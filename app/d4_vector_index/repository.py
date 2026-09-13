"""
repository.py — Async SQLAlchemy repository for D4 vector operations.

Session management
------------------
The module exposes a ``get_db()`` async generator intended for use as a
FastAPI ``Depends()`` dependency.  A new AsyncSession is created per request
and automatically rolled back on failure.

Engine selection
----------------
- USE_ALLOYDB=true  → Google AlloyDB Connector (asyncpg driver)
- Otherwise         → Direct URL from ASYNC_DATABASE_URL env var
                      (e.g. ``postgresql+asyncpg://user:pw@host:5432/db``)
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Optional

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings
from app.d4_vector_index.models import Chunk, ChunkVector

logger = logging.getLogger(__name__)

EMBEDDING_DIM: int = 768

# ---------------------------------------------------------------------------
# Lazy async engine + sessionmaker singletons
# ---------------------------------------------------------------------------

_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _build_engine():
    """
    Build the async SQLAlchemy engine based on runtime settings.
    Called once on first request; result is cached in module-level _engine.
    """
    s = get_settings()

    if s.use_alloydb:
        from google.cloud.alloydb.connector import AsyncConnector, IPTypes  # noqa: PLC0415

        ip_type_map = {"PUBLIC": IPTypes.PUBLIC, "PRIVATE": IPTypes.PRIVATE, "PSC": IPTypes.PSC}
        ip_type = ip_type_map.get(getattr(s, "alloydb_ip_type", "PUBLIC").upper(), IPTypes.PUBLIC)

        # Use a list so the closure can mutate it (avoids `nonlocal` keyword).
        _holder: list[AsyncConnector] = []

        async def _getconn():
            if not _holder:
                if s.alloydb_sa_key:
                    from google.oauth2 import service_account  # noqa: PLC0415

                    # Admin creds: AlloyDB instance metadata discovery
                    admin_creds = service_account.Credentials.from_service_account_file(
                        s.alloydb_sa_key,
                        scopes=["https://www.googleapis.com/auth/cloud-platform"],
                    )
                    # DB creds: IAM token exchange (alloydb.login scope only)
                    db_creds = service_account.Credentials.from_service_account_file(
                        s.alloydb_sa_key,
                        scopes=["https://www.googleapis.com/auth/alloydb.login"],
                    )
                    _holder.append(AsyncConnector(credentials=admin_creds, db_credentials=db_creds))
                else:
                    _holder.append(AsyncConnector())
            return await _holder[0].connect(
                s.alloydb_instance_uri,
                "asyncpg",
                user=s.alloydb_user,
                password=s.alloydb_password or None,
                db=s.alloydb_db,
                ip_type=ip_type,
                enable_iam_auth=True,
            )

        logger.info("D4: creating async engine via AlloyDB Connector (asyncpg)")
        return create_async_engine(
            "postgresql+asyncpg://",
            async_creator=_getconn,
            pool_size=5,
            max_overflow=2,
            pool_pre_ping=True,
            echo=False,
        )

    # Fallback: direct asyncpg URL (useful for local dev / integration tests)
    url = getattr(s, "async_database_url", "")
    if not url:
        raise RuntimeError(
            "D4 vector index requires either USE_ALLOYDB=true (with ALLOYDB_* "
            "env vars) or ASYNC_DATABASE_URL set to a postgresql+asyncpg:// URL."
        )
    logger.info("D4: creating async engine via ASYNC_DATABASE_URL")
    return create_async_engine(url, pool_pre_ping=True, echo=False)


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _engine, _session_factory
    if _session_factory is None:
        _engine = _build_engine()
        _session_factory = async_sessionmaker(
            _engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency: yields a scoped async SQLAlchemy session.

    Usage::

        @router.post("/d4/embed")
        async def embed(body: EmbedRequest, session: AsyncSession = Depends(get_db)):
            ...
    """
    factory = _get_session_factory()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class D4VectorRepository:
    """
    Async pgvector repository.

    All public methods are coroutines — never block the event loop.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # store_embedding
    # ------------------------------------------------------------------

    async def store_embedding(
        self,
        chunk_id: uuid.UUID,
        embedding: list[float],
        model: str,
    ) -> None:
        """
        Upsert the embedding for an existing chunk.

        Steps
        -----
        1. Validate embedding dimension == 768.
        2. Fetch chunk metadata from public.chunks (raises LookupError if not found).
        3. INSERT INTO chunk_vectors … ON CONFLICT DO UPDATE SET embedding = …
        4. Commit.

        Parameters
        ----------
        chunk_id:  UUID of an existing chunk (must exist in public.chunks).
        embedding: Flat list of 768 floats produced by an embedding model.
        model:     Name of the embedding model (recorded for traceability).
        """
        if len(embedding) != EMBEDDING_DIM:
            raise ValueError(
                f"Expected embedding dimension {EMBEDDING_DIM}, "
                f"got {len(embedding)}."
            )

        # 1. Fetch chunk metadata (needed to populate chunk_vectors columns)
        result = await self._session.execute(
            select(Chunk).where(Chunk.chunk_id == chunk_id)
        )
        chunk = result.scalar_one_or_none()
        if chunk is None:
            raise LookupError(
                f"chunk_id={chunk_id} not found in public.chunks. "
                "Ensure the chunk was ingested via D3 before embedding."
            )

        # 2. Upsert into chunk_vectors
        stmt = (
            pg_insert(ChunkVector)
            .values(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                jurisdiction=chunk.jurisdiction,
                doc_type=chunk.doc_type,
                trust_level=chunk.trust_level,
                effective_date=chunk.effective_date,
                source_org=chunk.source_org,
                embedding=embedding,
                embedding_model=model,
            )
            .on_conflict_do_update(
                index_elements=["chunk_id"],
                set_={
                    "embedding": embedding,
                    "embedding_model": model,
                    "embedded_at": text("NOW()"),
                },
            )
        )
        await self._session.execute(stmt)
        await self._session.commit()

    # ------------------------------------------------------------------
    # retrieve_top_k
    # ------------------------------------------------------------------

    async def retrieve_top_k(
        self,
        query_embedding: list[float],
        top_k: int = 8,
        jurisdiction: Optional[str] = None,
        doc_type: Optional[str] = None,
        language: Optional[str] = None,
        trust_max: Optional[int] = None,
    ) -> list[dict]:
        """
        Return top-k chunk IDs ranked by cosine similarity.

        Score formula
        -------------
        score = 1 − cosine_distance(cv.embedding, query_embedding)
              = 1 − (cv.embedding <=> query_embedding)

        A score of 1.0 means identical vectors; 0.0 means orthogonal.

        Parameters
        ----------
        query_embedding : 768-d query vector.
        top_k           : Maximum number of results to return (1–100).
        jurisdiction    : Filter to chunks from this jurisdiction.
        doc_type        : Filter to chunks of this document type.
        language        : Filter to chunks in this language
                          (requires a JOIN with public.chunks).
        trust_max       : Include only chunks whose trust_level <= trust_max.

        Returns
        -------
        List of dicts: [{"chunk_id": UUID, "score": float}, …]
        """
        # Serialise query vector as a pgvector literal string
        vec_str = "[" + ",".join(f"{v:.8f}" for v in query_embedding) + "]"

        # Build WHERE clauses and parameter dict dynamically
        where_parts: list[str] = ["cv.embedding IS NOT NULL"]
        params: dict = {"vec": vec_str, "top_k": top_k}

        # Language filter requires a JOIN with public.chunks (cv has no language)
        need_join = language is not None
        join_clause = (
            "JOIN public.chunks ch ON ch.chunk_id = cv.chunk_id"
            if need_join
            else ""
        )

        if jurisdiction is not None:
            where_parts.append("cv.jurisdiction = :jurisdiction")
            params["jurisdiction"] = jurisdiction

        if doc_type is not None:
            where_parts.append("cv.doc_type = :doc_type")
            params["doc_type"] = doc_type

        if trust_max is not None:
            where_parts.append("cv.trust_level <= :trust_max")
            params["trust_max"] = trust_max

        if language is not None:
            where_parts.append("ch.language = :language")
            params["language"] = language

        where_str = " AND ".join(where_parts)

        # Use (:vec)::vector — CAST(:vec AS vector) returns 0 rows with asyncpg/AlloyDB
        sql = text(
            f"""
            SELECT cv.chunk_id,
                   1 - (cv.embedding <=> (:vec)::vector) AS score
            FROM   public.chunk_vectors cv
            {join_clause}
            WHERE  {where_str}
            ORDER  BY cv.embedding <=> (:vec)::vector
            LIMIT  :top_k
            """
        )

        result = await self._session.execute(sql, params)
        rows = result.fetchall()
        return [
            {"chunk_id": row.chunk_id, "score": float(row.score)}
            for row in rows
        ]
