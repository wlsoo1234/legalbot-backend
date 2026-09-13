"""Backward-compatible consultation facade over the canonical RAG engine."""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.rag_chatbot.schemas import ChatRequest
from app.rag_chatbot.service import ask
from app.repositories.consultation_repository import (
    create_consultation_message,
    select_all_consultation_history,
)
from common.schemas import SendMessageRequest


async def get_consultation_history(session_id: str) -> dict:
    return {"items": await select_all_consultation_history(session_id)}


async def send_consultation_message(
    request: SendMessageRequest,
    db: AsyncSession,
) -> dict:
    session_id = request.session_id or str(uuid.uuid4())
    await create_consultation_message(request.text, "user", session_id)

    rag_response = await ask(
        ChatRequest(
            question=request.text,
            session_id=session_id,
            jurisdiction=request.jurisdiction,
            doc_type=request.doc_type,
            language=request.language,
            trust_level_max=request.trust_level_max,
            top_k=request.top_k,
        ),
        db,
    )
    display_text = rag_response.answer or (
        "I could not find enough reliable information in the indexed legal sources "
        "to answer that question."
    )
    message = await create_consultation_message(display_text, "model", session_id)
    return {
        **message,
        "sessionId": rag_response.session_id,
        "qaId": rag_response.qa_id,
        "citations": [citation.model_dump() for citation in rag_response.citations],
        "confidence": rag_response.confidence,
        "followUpQuestions": rag_response.follow_up_questions,
        "disclaimer": rag_response.disclaimer,
        "model": rag_response.model,
        "retrievalK": rag_response.retrieval_k,
    }
