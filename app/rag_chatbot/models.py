"""
models.py — SQLAlchemy 2.0 ORM models for Q&A persistence.

Maps to the live AlloyDB schema tables:
  public.qa_pairs     → QAPair      (one row per /chat/ask call)
  public.qa_citations → QACitation  (one row per cited chunk)

These tables are created by the SQL migration in app/db/schema.sql.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Float, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class QAPair(Base):
    """
    One row per question answered by the RAG chatbot.

    Columns
    -------
    qa_id             : Primary key (UUID).
    session_id        : Optional caller-supplied session/conversation ID.
    jurisdiction      : ISO 3166 code used to filter retrieval.
    question          : Original user question.
    answer            : Generated answer text (NULL if context was insufficient).
    confidence        : "high" | "low" | "none".
    model             : Gemini model name used for generation.
    retrieval_k       : Number of chunks retrieved from D4.
    retrieval_filters : JSONB bag of extra filters applied (doc_type, trust_level_max).
    created_at        : UTC timestamp.
    """

    __tablename__ = "qa_pairs"
    __table_args__ = {"schema": "public"}

    qa_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    jurisdiction: Mapped[str] = mapped_column(Text, nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[str | None] = mapped_column(Text, nullable=True)
    model: Mapped[str] = mapped_column(Text, nullable=False)
    retrieval_k: Mapped[int] = mapped_column(Integer, nullable=False)
    retrieval_filters: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        default=lambda: datetime.utcnow(),
    )


class QACitation(Base):
    """
    One row per chunk cited in a QAPair answer.

    Columns
    -------
    cit_id        : Primary key (UUID).
    qa_id         : FK → qa_pairs.qa_id (CASCADE DELETE).
    chunk_id      : UUID string of the cited chunk (TEXT for flexibility).
    doc_title     : Title of the source document.
    source_org    : Authoring organisation.
    page_start    : First page of the cited chunk (1-based).
    page_end      : Last page of the cited chunk.
    section_title : Section heading.
    link          : Canonical URL for the document.
    score         : Cosine similarity score from D4 retrieval (0–1).
    confidence    : "high" (chunk was in retrieved set) | "low" (hallucinated).
    created_at    : UTC timestamp.
    """

    __tablename__ = "qa_citations"
    __table_args__ = {"schema": "public"}

    cit_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    qa_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("public.qa_pairs.qa_id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    doc_title: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_org: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section_title: Mapped[str | None] = mapped_column(Text, nullable=True)
    link: Mapped[str | None] = mapped_column(Text, nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    retrieval_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        default=lambda: datetime.utcnow(),
    )
