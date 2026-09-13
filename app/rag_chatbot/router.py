"""
router.py — FastAPI router for the RAG Chatbot endpoint.

Mounts at prefix /api/v1/chat (registered in app/main.py).

Endpoints
---------
POST /api/v1/chat/ask
    Ask a legal question grounded in the indexed document corpus.
    Returns a structured ChatResponse with citations, confidence, and
    a mandatory legal disclaimer.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_async_session
from app.rag_chatbot import service
from app.rag_chatbot.schemas import ChatRequest, ChatResponse

router = APIRouter(
    prefix="/api/v1/chat",
    tags=["RAG Chatbot"],
)


@router.post(
    "/ask",
    response_model=ChatResponse,
    summary="Ask a legal question",
    description=(
        "Retrieves legal document chunks with hybrid vector and full-text search, "
        "then calls Gemini with strict JSON output to produce a grounded answer "
        "with citations.  The Q&A pair and citations are persisted to AlloyDB."
    ),
)
async def chat_ask(
    body: ChatRequest,
    session: AsyncSession = Depends(get_async_session),
) -> ChatResponse:
    """
    Full RAG chatbot pipeline:

    1. Resolve follow-up questions from recent session history.
    2. Retrieve top-k chunks using pgvector + PostgreSQL full-text search.
    3. Fetch chunk text + metadata from AlloyDB.
    4. Call Gemini (gemini-2.5-flash) with strict JSON schema.
    5. Validate citations and build their metadata entirely server-side.
    6. Persist qa_pairs + qa_citations atomically.
    7. Return ChatResponse.
    """
    return await service.ask(body, session)
