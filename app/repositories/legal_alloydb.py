"""
repositories/legal_alloydb.py — AlloyDB + pgvector implementations of
LegalKnowledgeRepository and VectorIndexRepository.

These are drop-in replacements for the in-memory versions in legal_memory.py.
Swap them in via deps_rag.py by setting USE_ALLOYDB=true in .env — no router
or service code changes required.

Prerequisites
-------------
1. Run app/db/schema.sql against your AlloyDB instance once.
2. Set these env vars in .env:
     ALLOYDB_INSTANCE_URI  projects/<P>/locations/<R>/clusters/<C>/instances/<I>
     ALLOYDB_DB, ALLOYDB_USER, ALLOYDB_PASSWORD
     USE_ALLOYDB=true

pg8000 notes
------------
- pg8000 uses positional %s parameters only (paramstyle='format').
  Named %(key)s params are NOT supported — all queries use %s + tuples/lists.
- pg8000 cursors do NOT support the context-manager protocol (no __enter__).
  Every cursor is created with `cur = conn.cursor()` and closed in a finally block.
- pg8000 returns plain tuples; _dict_row() / _dict_rows() convert them to dicts
  using cursor.description for column names.
- psycopg2.extras (RealDictCursor, execute_batch) must NOT be used on pg8000
  connections.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.db.alloydb import get_connection
from app.repositories.legal_base import LegalKnowledgeRepository, VectorIndexRepository
from app.schemas_rag import (
    ChunkRecord,
    LegalDocument,
    LegalSourceCreate,
    RegisteredSource,
    SourceRegisterRequest,
    SourceUpdateRequest,
)


# ---------------------------------------------------------------------------
# pg8000 row helpers — build dicts from cursor.description
# ---------------------------------------------------------------------------

def _dict_row(cur) -> dict | None:
    """Fetch one row as a {column: value} dict, or None if no row."""
    row = cur.fetchone()
    if row is None:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))


def _dict_rows(cur) -> list[dict]:
    """Fetch all rows as a list of {column: value} dicts."""
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Row -> Pydantic model helpers
# ---------------------------------------------------------------------------

def _row_to_registered_source(row: dict) -> RegisteredSource:
    return RegisteredSource(
        source_id=str(row["source_id"]),
        source_type=row["source_type"],
        canonical_url=row.get("canonical_url"),
        title=row.get("title"),
        source_org=row.get("source_org"),
        trust_level=int(row["trust_level"]),
        jurisdiction=row["jurisdiction"],
        doc_type=row["doc_type"],
        language=row.get("language", "en"),
        effective_date=row.get("effective_date"),
        tags=list(row.get("tags") or []),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        status=row.get("status", "registered"),
    )


def _row_to_legal_document(row: dict) -> LegalDocument:
    return LegalDocument(
        doc_id=str(row["doc_id"]),
        source_id=str(row["source_id"]) if row.get("source_id") else None,
        provider=row.get("provider"),
        title=row.get("title"),
        url=row.get("url"),
        jurisdiction=row["jurisdiction"],
        doc_type=row["doc_type"],
        created_at=row.get("created_at"),
    )


def _row_to_chunk_record(row: dict) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=str(row["chunk_id"]),
        doc_id=str(row["doc_id"]),
        chunk_index=int(row.get("chunk_index", 0)),
        text=row["text"],
        token_estimate=row.get("token_estimate"),
        page_start=row.get("page_start"),
        page_end=row.get("page_end"),
        section_title=row.get("section_title"),
        url_fragment=row.get("url_fragment"),
        char_start=row.get("char_start"),
        char_end=row.get("char_end"),
        jurisdiction=row.get("jurisdiction", ""),
        doc_type=row.get("doc_type", ""),
        effective_date=row.get("effective_date"),
        language=row.get("language", "en"),
        source_org=row.get("source_org"),
        trust_level=int(row.get("trust_level", 3)),
        tags=list(row.get("tags") or []),
        version_label=row.get("version_label"),
        created_at=row["created_at"],
    )


# ---------------------------------------------------------------------------
# D3 — AlloyDB LegalKnowledgeRepository
# ---------------------------------------------------------------------------

class AlloyDBKnowledgeRepository(LegalKnowledgeRepository):
    """
    Persists legal sources, documents, and chunks in AlloyDB (PostgreSQL).
    """

    # ------------------------------------------------------------------
    # Source registry
    # ------------------------------------------------------------------

    def register_source(self, request: SourceRegisterRequest) -> RegisteredSource:
        source_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        with get_connection() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    """
                    INSERT INTO legal_sources (
                        source_id, source_type, canonical_url, title, source_org,
                        trust_level, jurisdiction, doc_type, language, effective_date,
                        tags, status, created_at, updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'registered', %s, %s
                    )
                    RETURNING *
                    """,
                    (
                        source_id,
                        request.source_type,
                        request.canonical_url,
                        request.title,
                        request.source_org,
                        request.trust_level,
                        request.jurisdiction,
                        request.doc_type,
                        request.language,
                        request.effective_date,
                        list(request.tags),
                        now,
                        now,
                    ),
                )
                row = _dict_row(cur)
            finally:
                cur.close()
        return _row_to_registered_source(row)

    def get_source(self, source_id: str) -> RegisteredSource | None:
        with get_connection() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    "SELECT * FROM legal_sources WHERE source_id = %s",
                    (source_id,),
                )
                row = _dict_row(cur)
            finally:
                cur.close()
        return _row_to_registered_source(row) if row else None

    def update_source(
        self, source_id: str, updates: SourceUpdateRequest
    ) -> RegisteredSource | None:
        patch = {k: v for k, v in updates.model_dump().items() if v is not None}
        if not patch:
            return self.get_source(source_id)

        patch["updated_at"] = datetime.now(timezone.utc)
        cols = list(patch.keys())
        set_clause = ", ".join(f"{col} = %s" for col in cols)
        values = [patch[col] for col in cols] + [source_id]

        with get_connection() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    f"UPDATE legal_sources SET {set_clause} "
                    f"WHERE source_id = %s RETURNING *",
                    values,
                )
                row = _dict_row(cur)
            finally:
                cur.close()
        return _row_to_registered_source(row) if row else None

    def list_sources(
        self,
        jurisdiction: str | None = None,
        doc_type: str | None = None,
        status: str | None = None,
    ) -> list[RegisteredSource]:
        clauses = ["1=1"]
        params: list = []
        if jurisdiction:
            clauses.append("jurisdiction = %s")
            params.append(jurisdiction)
        if doc_type:
            clauses.append("doc_type = %s")
            params.append(doc_type)
        if status is not None:
            clauses.append("status = %s")
            params.append(status)

        where = " AND ".join(clauses)
        with get_connection() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    f"SELECT * FROM legal_sources WHERE {where} ORDER BY created_at DESC",
                    params,
                )
                rows = _dict_rows(cur)
            finally:
                cur.close()
        return [_row_to_registered_source(r) for r in rows]

    def link_doc_to_source(self, source_id: str, doc_id: str) -> None:
        with get_connection() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    """
                    UPDATE legal_sources
                       SET status = 'ingested',
                           updated_at = NOW()
                     WHERE source_id = %s
                    """,
                    (source_id,),
                )
            finally:
                cur.close()

    # ------------------------------------------------------------------
    # Document + chunk store
    # ------------------------------------------------------------------

    def upsert_document(
        self,
        source: LegalSourceCreate,
        text: str,
    ) -> tuple[LegalDocument, list[ChunkRecord]]:
        """
        Create or replace a legal_documents row and chunk the provided text.

        Idempotent: if a document with the same (provider, title, jurisdiction)
        already exists its stale chunks/vectors are deleted before re-ingesting.
        """
        from app.repositories.legal_memory import _chunk_text  # noqa: PLC0415

        now = datetime.now(timezone.utc)
        with get_connection() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    """
                    SELECT doc_id FROM legal_documents
                     WHERE provider = %s AND title = %s AND jurisdiction = %s
                    """,
                    (source.provider, source.title, source.jurisdiction),
                )
                existing = _dict_row(cur)

                if existing:
                    doc_id = str(existing["doc_id"])
                    cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
                    cur.execute("DELETE FROM chunk_vectors WHERE doc_id = %s", (doc_id,))
                    cur.execute(
                        "UPDATE legal_documents SET url = %s WHERE doc_id = %s",
                        (source.url, doc_id),
                    )
                else:
                    doc_id = str(uuid.uuid4())
                    cur.execute(
                        """
                        INSERT INTO legal_documents
                          (doc_id, source_id, provider, title, url,
                           jurisdiction, doc_type, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            doc_id,
                            None,
                            source.provider,
                            source.title,
                            source.url,
                            source.jurisdiction,
                            source.doc_type,
                            now,
                        ),
                    )
            finally:
                cur.close()

        doc = LegalDocument(
            doc_id=doc_id,
            source_id=None,
            provider=source.provider,
            title=source.title,
            url=source.url,
            jurisdiction=source.jurisdiction,
            doc_type=source.doc_type,
            created_at=now,
        )

        chunks = _chunk_text(text, doc_id, source)
        if chunks:
            self.upsert_chunks(chunks)
        return doc, chunks

    def upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        """Persist a batch of ChunkRecords to the `chunks` table.

        Uses a single multi-row INSERT per call (one network round-trip)
        instead of one execute per row, which is critical for remote AlloyDB.
        """
        if not chunks:
            return

        # Build one INSERT with N value rows — single round-trip regardless of batch size
        cols = (
            "chunk_id, doc_id, chunk_index, text, token_estimate, "
            "page_start, page_end, section_title, url_fragment, "
            "char_start, char_end, jurisdiction, doc_type, "
            "effective_date, language, source_org, trust_level, "
            "tags, version_label, created_at"
        )
        row_placeholder = "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        placeholders = ", ".join(row_placeholder for _ in chunks)
        sql = (
            f"INSERT INTO chunks ({cols}) VALUES {placeholders} "
            "ON CONFLICT (chunk_id) DO NOTHING"
        )
        params: list = []
        for c in chunks:
            params.extend([
                c.chunk_id, c.doc_id, c.chunk_index, c.text, c.token_estimate,
                c.page_start, c.page_end, c.section_title, c.url_fragment,
                c.char_start, c.char_end, c.jurisdiction, c.doc_type,
                c.effective_date, c.language, c.source_org, c.trust_level,
                list(c.tags), c.version_label, c.created_at,
            ])

        with get_connection() as conn:
            cur = conn.cursor()
            try:
                cur.execute(sql, params)
            finally:
                cur.close()

    def get_document(self, doc_id: str) -> LegalDocument | None:
        with get_connection() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    "SELECT * FROM legal_documents WHERE doc_id = %s", (doc_id,)
                )
                row = _dict_row(cur)
            finally:
                cur.close()
        return _row_to_legal_document(row) if row else None

    def save_document(self, doc: LegalDocument) -> None:
        """Persist a pre-built LegalDocument row (idempotent — no-op on duplicate doc_id)."""
        with get_connection() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    """
                    INSERT INTO legal_documents
                      (doc_id, source_id, provider, title, url,
                       jurisdiction, doc_type, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (doc_id) DO NOTHING
                    """,
                    (
                        doc.doc_id,
                        doc.source_id,
                        doc.provider,
                        doc.title,
                        doc.url,
                        doc.jurisdiction,
                        doc.doc_type,
                        doc.created_at,
                    ),
                )
            finally:
                cur.close()

    def get_chunks(self, chunk_ids: list[str]) -> list[ChunkRecord]:
        if not chunk_ids:
            return []
        with get_connection() as conn:
            cur = conn.cursor()
            try:
                placeholders = ", ".join(["%s"] * len(chunk_ids))
                cur.execute(
                    f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
                    chunk_ids,
                )
                rows = _dict_rows(cur)
            finally:
                cur.close()
        row_map = {str(r["chunk_id"]): _row_to_chunk_record(r) for r in rows}
        return [row_map[cid] for cid in chunk_ids if cid in row_map]


# ---------------------------------------------------------------------------
# D4 — AlloyDB pgvector VectorIndexRepository
# ---------------------------------------------------------------------------

class AlloyDBVectorRepository(VectorIndexRepository):
    """
    Stores embeddings in the ``chunk_vectors`` table using the pgvector
    ``vector`` type.  Uses cosine distance (``<=>`` operator) for retrieval.
    """

    def upsert_vectors(
        self, items: list[tuple[str, list[float], dict]]
    ) -> None:
        """Insert or replace (chunk_id, embedding, metadata) triples."""
        if not items:
            return
        with get_connection() as conn:
            cur = conn.cursor()
            try:
                for chunk_id, embedding, meta in items:
                    emb_str = "[" + ",".join(str(v) for v in embedding) + "]"
                    cur.execute(
                        """
                        INSERT INTO chunk_vectors
                          (chunk_id, doc_id, jurisdiction, doc_type,
                           trust_level, effective_date, source_org, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                        ON CONFLICT (chunk_id) DO UPDATE
                            SET embedding      = EXCLUDED.embedding,
                                doc_id         = EXCLUDED.doc_id,
                                jurisdiction   = EXCLUDED.jurisdiction,
                                doc_type       = EXCLUDED.doc_type,
                                trust_level    = EXCLUDED.trust_level,
                                effective_date = EXCLUDED.effective_date,
                                source_org     = EXCLUDED.source_org
                        """,
                        (
                            chunk_id,
                            meta.get("doc_id", ""),
                            meta.get("jurisdiction", ""),
                            meta.get("doc_type", ""),
                            meta.get("trust_level", 3),
                            meta.get("effective_date"),
                            meta.get("source_org"),
                            emb_str,
                        ),
                    )
            finally:
                cur.close()

    def query(
        self,
        embedding: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[tuple[str, float]]:
        """
        Retrieve top-k chunk IDs by cosine similarity with optional filters.

        Supported filter keys: jurisdiction, doc_type, doc_id,
        trust_level_max, effective_date_min.
        """
        emb_str = "[" + ",".join(str(v) for v in embedding) + "]"
        clauses = ["1=1"]
        params: list = []

        if filters:
            if filters.get("jurisdiction"):
                clauses.append("jurisdiction = %s")
                params.append(filters["jurisdiction"])
            if filters.get("doc_type"):
                clauses.append("doc_type = %s")
                params.append(filters["doc_type"])
            if filters.get("doc_id"):
                clauses.append("doc_id = %s")
                params.append(filters["doc_id"])
            if filters.get("trust_level_max") is not None:
                clauses.append("trust_level <= %s")
                params.append(filters["trust_level_max"])
            if filters.get("effective_date_min") is not None:
                clauses.append(
                    "(effective_date IS NULL OR effective_date >= %s)"
                )
                params.append(filters["effective_date_min"])

        where = " AND ".join(clauses)
        # emb_str appears twice: SELECT score + ORDER BY
        sql = f"""
            SELECT chunk_id,
                   1 - (embedding <=> %s::vector) AS score
              FROM chunk_vectors
             WHERE {where}
             ORDER BY embedding <=> %s::vector
             LIMIT %s
        """
        query_params = [emb_str] + params + [emb_str, top_k]

        with get_connection() as conn:
            cur = conn.cursor()
            try:
                cur.execute(sql, query_params)
                rows = _dict_rows(cur)
            finally:
                cur.close()
        return [(str(r["chunk_id"]), float(r["score"])) for r in rows]
