"""
service.py — Business logic layer for the D4 vector index.

Sits between the router (HTTP concerns) and the repository (database concerns).
Validates inputs and transforms repository results into Pydantic response models.
"""

from __future__ import annotations

import uuid
import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.d4_vector_index.repository import D4VectorRepository
from app.d4_vector_index.schemas import EMBEDDING_DIM, RetrieveResult

logger = logging.getLogger(__name__)


class D4VectorService:
    """
    Stateless service class — a new instance is created per request.

    Parameters
    ----------
    session : AsyncSession injected by the FastAPI ``get_db`` dependency.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._repo = D4VectorRepository(session)

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
        Validate and persist an embedding for a chunk.

        Raises
        ------
        ValueError   : embedding dimension != 768.
        LookupError  : chunk_id does not exist in public.chunks.
        """
        if len(embedding) != EMBEDDING_DIM:
            raise ValueError(
                f"embedding must have exactly {EMBEDDING_DIM} dimensions, "
                f"got {len(embedding)}."
            )
        logger.debug(
            "store_embedding chunk_id=%s model=%s dim=%d",
            chunk_id,
            model,
            len(embedding),
        )
        await self._repo.store_embedding(chunk_id, embedding, model)

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
    ) -> list[RetrieveResult]:
        """
        Return top-k similar chunks by cosine similarity.

        Parameters
        ----------
        query_embedding : 768-d query vector.
        top_k           : Max results (1–100, default 8).
        jurisdiction    : Optional jurisdiction filter.
        doc_type        : Optional document-type filter.
        language        : Optional language filter (BCP-47 tag, e.g. ``"en"``).
        trust_max       : Optional trust-level ceiling (1–5).

        Returns
        -------
        List of RetrieveResult sorted by descending score.
        """
        if len(query_embedding) != EMBEDDING_DIM:
            raise ValueError(
                f"query_embedding must have exactly {EMBEDDING_DIM} dimensions, "
                f"got {len(query_embedding)}."
            )
        logger.debug(
            "retrieve_top_k top_k=%d jurisdiction=%s doc_type=%s "
            "language=%s trust_max=%s",
            top_k,
            jurisdiction,
            doc_type,
            language,
            trust_max,
        )
        raw = await self._repo.retrieve_top_k(
            query_embedding=query_embedding,
            top_k=top_k,
            jurisdiction=jurisdiction,
            doc_type=doc_type,
            language=language,
            trust_max=trust_max,
        )
        return [
            RetrieveResult(chunk_id=row["chunk_id"], score=row["score"])
            for row in raw
        ]
