"""Async persistence and read models for canonical RAG conversations."""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.rag_chatbot.schemas import CitationOut, ConversationTurn


async def fetch_recent_turns(
    session: AsyncSession,
    session_id: str,
    *,
    limit: int,
) -> list[ConversationTurn]:
    result = await session.execute(
        text(
            """
            SELECT question, answer
              FROM public.qa_pairs
             WHERE session_id = :session_id
             ORDER BY created_at DESC
             LIMIT :turn_limit
            """
        ),
        {"session_id": session_id, "turn_limit": limit},
    )
    rows = list(reversed(result.mappings().all()))
    return [ConversationTurn(question=row["question"], answer=row["answer"]) for row in rows]


async def fetch_chunk_details(
    session: AsyncSession,
    chunk_ids: list[str],
) -> dict[str, dict]:
    if not chunk_ids:
        return {}
    result = await session.execute(
        text(
            """
            SELECT c.chunk_id::text AS chunk_id,
                   c.text,
                   c.page_start,
                   c.page_end,
                   c.section_title,
                   c.source_org,
                   COALESCE(ld.title, ls.title, 'Unknown Document') AS doc_title,
                   COALESCE(ld.url, ls.canonical_url) AS link
              FROM public.chunks c
              LEFT JOIN public.legal_documents ld ON ld.doc_id = c.doc_id
              LEFT JOIN public.documents d ON d.doc_id = c.doc_id
              LEFT JOIN public.legal_sources ls ON ls.source_id = d.source_id
             WHERE c.chunk_id = ANY(CAST(:chunk_ids AS uuid[]))
            """
        ),
        {"chunk_ids": chunk_ids},
    )
    return {row["chunk_id"]: dict(row) for row in result.mappings().all()}


async def save_exchange(
    session: AsyncSession,
    *,
    qa_id: uuid.UUID,
    session_id: str,
    jurisdiction: str,
    question: str,
    answer: str | None,
    confidence: str,
    model: str,
    retrieval_k: int,
    retrieval_filters: dict | None,
    citations: Sequence[CitationOut],
) -> None:
    """Persist a Q&A row and all citations with one commit."""
    try:
        await session.execute(
            text(
                """
                INSERT INTO public.qa_pairs
                    (qa_id, session_id, jurisdiction, question, answer,
                     confidence, model, retrieval_k, retrieval_filters, created_at)
                VALUES
                    (:qa_id, :session_id, :jurisdiction, :question, :answer,
                     :confidence, :model, :retrieval_k,
                     CAST(:retrieval_filters AS jsonb), NOW())
                """
            ),
            {
                "qa_id": str(qa_id),
                "session_id": session_id,
                "jurisdiction": jurisdiction,
                "question": question,
                "answer": answer,
                "confidence": confidence,
                "model": model,
                "retrieval_k": retrieval_k,
                "retrieval_filters": json.dumps(retrieval_filters or {}),
            },
        )
        for citation in citations:
            await session.execute(
                text(
                    """
                    INSERT INTO public.qa_citations
                        (cit_id, qa_id, chunk_id, doc_title, source_org,
                         page_start, page_end, section_title, link,
                         score, retrieval_score, confidence, created_at)
                    VALUES
                        (:cit_id, :qa_id, :chunk_id, :doc_title, :source_org,
                         :page_start, :page_end, :section_title, :link,
                         :score, :retrieval_score, :confidence, NOW())
                    """
                ),
                {
                    "cit_id": str(uuid.uuid4()),
                    "qa_id": str(qa_id),
                    **citation.model_dump(),
                },
            )
        await session.commit()
    except Exception:
        await session.rollback()
        raise
