"""Public and model-facing schemas for the canonical RAG chatbot."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = Field(None, min_length=1, max_length=128)
    jurisdiction: str = Field("MY", min_length=2, max_length=32)
    top_k: int = Field(5, ge=1, le=20)
    doc_type: str | None = Field(None, max_length=100)
    language: str | None = Field(None, max_length=32)
    trust_level_max: int | None = Field(None, ge=1, le=5)

    @field_validator("question", "jurisdiction")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class QueryRewriteJSON(BaseModel):
    search_query: str = Field(..., min_length=1, max_length=4000)


class AnswerJSON(BaseModel):
    """Minimal structured output accepted from Gemini."""

    answer: str | None = Field(None, max_length=8000)
    cited_chunk_ids: list[str] = Field(default_factory=list)
    confidence: Literal["high", "low", "none"] = "low"
    follow_up_questions: list[str] = Field(default_factory=list, max_length=3)


class CitationOut(BaseModel):
    """Citation metadata is populated by the server, never by the model."""

    chunk_id: str
    doc_title: str
    source_org: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    section_title: str | None = None
    link: str | None = None
    score: float | None = Field(None, description="Cosine similarity when available.")
    retrieval_score: float = Field(..., ge=0.0, le=1.0)
    confidence: Literal["high"] = "high"


class ChatResponse(BaseModel):
    qa_id: str
    session_id: str
    answer: str | None = None
    citations: list[CitationOut] = Field(default_factory=list)
    confidence: Literal["high", "low", "none"]
    follow_up_questions: list[str] = Field(default_factory=list)
    disclaimer: str
    model: str
    retrieval_k: int


class ConversationTurn(BaseModel):
    question: str
    answer: str | None = None
