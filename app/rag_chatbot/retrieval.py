"""Hybrid pgvector + PostgreSQL full-text retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from app.logging_context import get_request_id

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HybridCandidate:
    chunk_id: str
    retrieval_score: float
    vector_score: float | None = None
    lexical_score: float | None = None
    matched_by: tuple[str, ...] = ()


def reciprocal_rank_fusion(
    vector_rows: list[dict],
    lexical_rows: list[dict],
    *,
    rrf_k: int = 60,
    limit: int = 5,
) -> list[HybridCandidate]:
    """Fuse two ranked lists deterministically and normalize the top score to 1."""
    fused: dict[str, dict] = {}
    for source, rows in (("vector", vector_rows), ("lexical", lexical_rows)):
        seen_in_source: set[str] = set()
        for rank, row in enumerate(rows, start=1):
            chunk_id = str(row["chunk_id"])
            if chunk_id in seen_in_source:
                continue
            seen_in_source.add(chunk_id)
            item = fused.setdefault(
                chunk_id,
                {
                    "raw_score": 0.0,
                    "vector_score": None,
                    "lexical_score": None,
                    "matched_by": [],
                },
            )
            item["raw_score"] += 1.0 / (rrf_k + rank)
            if source not in item["matched_by"]:
                item["matched_by"].append(source)
            if row.get("vector_score") is not None:
                item["vector_score"] = float(row["vector_score"])
            if row.get("lexical_score") is not None:
                item["lexical_score"] = float(row["lexical_score"])

    ordered = sorted(
        fused.items(),
        key=lambda pair: (-pair[1]["raw_score"], pair[0]),
    )[:limit]
    max_score = ordered[0][1]["raw_score"] if ordered else 1.0
    return [
        HybridCandidate(
            chunk_id=chunk_id,
            retrieval_score=min(1.0, values["raw_score"] / max_score),
            vector_score=values["vector_score"],
            lexical_score=values["lexical_score"],
            matched_by=tuple(values["matched_by"]),
        )
        for chunk_id, values in ordered
    ]


def _filters(
    jurisdiction: str,
    doc_type: str | None,
    language: str | None,
    trust_level_max: int | None,
) -> tuple[str, dict]:
    clauses = ["c.jurisdiction = :jurisdiction"]
    params: dict = {"jurisdiction": jurisdiction}
    if doc_type:
        clauses.append("c.doc_type = :doc_type")
        params["doc_type"] = doc_type
    if language:
        clauses.append("c.language = :language")
        params["language"] = language
    if trust_level_max is not None:
        clauses.append("c.trust_level <= :trust_level_max")
        params["trust_level_max"] = trust_level_max
    return " AND ".join(clauses), params


async def retrieve_hybrid(
    session: AsyncSession,
    *,
    query_embedding: list[float],
    query_text: str,
    embedding_model: str,
    jurisdiction: str,
    doc_type: str | None,
    language: str | None,
    trust_level_max: int | None,
    vector_limit: int,
    lexical_limit: int,
    final_limit: int,
    rrf_k: int,
) -> list[HybridCandidate]:
    where_sql, filter_params = _filters(
        jurisdiction,
        doc_type,
        language,
        trust_level_max,
    )
    vector_literal = "[" + ",".join(f"{value:.8f}" for value in query_embedding) + "]"

    vector_result = await session.execute(
        text(
            f"""
            SELECT c.chunk_id::text AS chunk_id,
                   1 - (cv.embedding <=> (:vector)::vector) AS vector_score,
                   NULL::float AS lexical_score
              FROM public.chunk_vectors cv
              JOIN public.chunks c ON c.chunk_id = cv.chunk_id
              JOIN public.documents d ON d.doc_id = c.doc_id AND d.status = 'ready'
             WHERE cv.embedding IS NOT NULL
               AND cv.embedding_model = :embedding_model
               AND {where_sql}
             ORDER BY cv.embedding <=> (:vector)::vector
             LIMIT :candidate_limit
            """
        ),
        {
            **filter_params,
            "vector": vector_literal,
            "embedding_model": embedding_model,
            "candidate_limit": vector_limit,
        },
    )
    vector_rows = [dict(row) for row in vector_result.mappings().all()]

    lexical_result = await session.execute(
        text(
            f"""
            WITH query AS (
                SELECT websearch_to_tsquery('simple', :query_text) AS value
            )
            SELECT c.chunk_id::text AS chunk_id,
                   NULL::float AS vector_score,
                   ts_rank_cd(c.search_vector, query.value) AS lexical_score
              FROM public.chunks c
              JOIN public.documents d ON d.doc_id = c.doc_id AND d.status = 'ready'
              CROSS JOIN query
             WHERE c.search_vector @@ query.value
               AND {where_sql}
             ORDER BY lexical_score DESC, c.chunk_id
             LIMIT :candidate_limit
            """
        ),
        {
            **filter_params,
            "query_text": query_text,
            "candidate_limit": lexical_limit,
        },
    )
    lexical_rows = [dict(row) for row in lexical_result.mappings().all()]

    fused = reciprocal_rank_fusion(
        vector_rows,
        lexical_rows,
        rrf_k=rrf_k,
        limit=final_limit,
    )
    logger.info(
        "rag_retrieval request_id=%s vector_candidates=%d lexical_candidates=%d fused_candidates=%d",
        get_request_id(),
        len(vector_rows),
        len(lexical_rows),
        len(fused),
    )
    return fused
