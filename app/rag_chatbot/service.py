"""Canonical hybrid, multi-turn and citation-grounded RAG orchestration."""

from __future__ import annotations

import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.d4_vector_index.embeddings import embed_text
from app.rag_chatbot import gemini_client, repository
from app.rag_chatbot.prompts import (
    DISCLAIMER,
    REWRITE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_context_block,
    build_rewrite_prompt,
    build_user_prompt,
    format_history,
)
from app.rag_chatbot.retrieval import HybridCandidate, retrieve_hybrid
from app.rag_chatbot.schemas import ChatRequest, ChatResponse, CitationOut
from app.logging_context import get_request_id

logger = logging.getLogger(__name__)


async def _standalone_query(request: ChatRequest, history) -> str:
    if not history:
        return request.question
    try:
        rewritten = await gemini_client.rewrite_query(
            REWRITE_SYSTEM_PROMPT,
            build_rewrite_prompt(request.question, history),
        )
        return rewritten or request.question
    except Exception as exc:
        logger.warning("rag_query_rewrite_failed session_id=%s error=%s", request.session_id, exc)
        return request.question


def _is_low_context(candidates: list[HybridCandidate], threshold: float) -> bool:
    if not candidates:
        return True
    has_lexical_match = any("lexical" in candidate.matched_by for candidate in candidates)
    best_vector = max(
        (candidate.vector_score for candidate in candidates if candidate.vector_score is not None),
        default=-1.0,
    )
    return not has_lexical_match and best_vector < threshold


def _server_citations(
    cited_chunk_ids: list[str],
    candidates: list[HybridCandidate],
    details: dict[str, dict],
) -> list[CitationOut]:
    candidate_by_id = {candidate.chunk_id: candidate for candidate in candidates}
    citations: list[CitationOut] = []
    seen: set[str] = set()
    for chunk_id in cited_chunk_ids:
        if chunk_id in seen or chunk_id not in candidate_by_id or chunk_id not in details:
            continue
        seen.add(chunk_id)
        candidate = candidate_by_id[chunk_id]
        detail = details[chunk_id]
        citations.append(
            CitationOut(
                chunk_id=chunk_id,
                doc_title=detail.get("doc_title") or "Unknown Document",
                source_org=detail.get("source_org"),
                page_start=detail.get("page_start"),
                page_end=detail.get("page_end"),
                section_title=detail.get("section_title"),
                link=detail.get("link"),
                score=candidate.vector_score,
                retrieval_score=candidate.retrieval_score,
            )
        )
    return citations


async def ask(request: ChatRequest, session: AsyncSession) -> ChatResponse:
    settings = get_settings()
    if not settings.rag_enabled:
        raise RuntimeError("RAG is disabled by configuration.")

    session_id = request.session_id or str(uuid.uuid4())
    request = request.model_copy(update={"session_id": session_id})
    history = await repository.fetch_recent_turns(
        session,
        session_id,
        limit=settings.rag_history_turns,
    )
    search_query = await _standalone_query(request, history)
    query_vector = await embed_text(search_query, task_type="RETRIEVAL_QUERY")

    candidates = await retrieve_hybrid(
        session,
        query_embedding=query_vector,
        query_text=search_query,
        embedding_model=settings.rag_embedding_model,
        jurisdiction=request.jurisdiction,
        doc_type=request.doc_type,
        language=request.language,
        trust_level_max=request.trust_level_max,
        vector_limit=settings.rag_vector_candidates,
        lexical_limit=settings.rag_lexical_candidates,
        final_limit=request.top_k,
        rrf_k=settings.rag_rrf_k,
    )
    details = await repository.fetch_chunk_details(
        session,
        [candidate.chunk_id for candidate in candidates],
    )
    ordered_details = [
        details[candidate.chunk_id]
        for candidate in candidates
        if candidate.chunk_id in details
    ]
    low_context = _is_low_context(candidates, settings.rag_low_context_threshold)

    if ordered_details:
        dynamic_overhead = len(request.question) + len(format_history(history))
        context_budget = max(0, settings.rag_max_context_chars - dynamic_overhead)
        context = build_context_block(
            ordered_details,
            max_chars=context_budget,
        )
        generated = await gemini_client.ask(
            SYSTEM_PROMPT,
            build_user_prompt(
                request.question,
                context,
                history,
                low_context=low_context,
            ),
        )
        citations = _server_citations(generated.cited_chunk_ids, candidates, details)
        answer = generated.answer
        confidence = generated.confidence
        follow_ups = generated.follow_up_questions
        if answer is None or not answer.strip() or confidence == "none":
            answer = None
            citations = []
            confidence = "none"
        elif not citations:
            answer = None
            confidence = "none"
            follow_ups = follow_ups or [
                "Could you clarify which tenancy issue or legal document you mean?",
                "Do you know the relevant clause, date, or authority?",
            ]
        elif low_context and confidence == "high":
            confidence = "low"
    else:
        answer = None
        citations = []
        confidence = "none"
        follow_ups = [
            "Could you provide more detail about the tenancy issue?",
            "Is there a particular Malaysian law or agreement clause you want to check?",
        ]

    if answer is None and not follow_ups:
        follow_ups = [
            "Could you provide more detail about the tenancy issue?",
            "Do you know the relevant clause, date, or legal authority?",
        ]

    qa_id = uuid.uuid4()
    filters = {
        "doc_type": request.doc_type,
        "language": request.language,
        "trust_level_max": request.trust_level_max,
        "search_query": search_query,
        "retrieval": "hybrid_rrf",
    }
    await repository.save_exchange(
        session,
        qa_id=qa_id,
        session_id=session_id,
        jurisdiction=request.jurisdiction,
        question=request.question,
        answer=answer,
        confidence=confidence,
        model=settings.rag_chat_model,
        retrieval_k=len(ordered_details),
        retrieval_filters=filters,
        citations=citations,
    )

    logger.info(
        "rag_answer request_id=%s qa_id=%s session_id=%s candidates=%d citations=%d confidence=%s",
        get_request_id(),
        qa_id,
        session_id,
        len(candidates),
        len(citations),
        confidence,
    )
    return ChatResponse(
        qa_id=str(qa_id),
        session_id=session_id,
        answer=answer,
        citations=citations,
        confidence=confidence,
        follow_up_questions=follow_ups,
        disclaimer=DISCLAIMER,
        model=settings.rag_chat_model,
        retrieval_k=len(ordered_details),
    )
