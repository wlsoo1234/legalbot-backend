"""
SQLAlchemy 2.0 ORM models for the D4 vector index.

Maps to the live AlloyDB schema tables:
  public.chunks        → Chunk         (D3 source; read-only from D4)
  public.chunk_vectors → ChunkVector   (D4 embedding store)

SQL migration (idempotent — run once against AlloyDB before using D4):
---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS vector;

-- This table is created by D3 (schema.sql).  D4 only populates `embedding`.
CREATE TABLE IF NOT EXISTS public.chunk_vectors (
    chunk_id       UUID     PRIMARY KEY
                            REFERENCES public.chunks(chunk_id) ON DELETE CASCADE,
    doc_id         UUID     NOT NULL,
    jurisdiction   TEXT     NOT NULL,
    doc_type       TEXT     NOT NULL,
    trust_level    SMALLINT NOT NULL DEFAULT 3,
    effective_date DATE,
    source_org     TEXT,
    embedding      vector(768)          -- nullable; populated by POST /d4/embed
);

CREATE INDEX IF NOT EXISTS chunk_vectors_embedding_idx
    ON public.chunk_vectors
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_chunk_vectors_jurisdiction
    ON public.chunk_vectors (jurisdiction);
CREATE INDEX IF NOT EXISTS idx_chunk_vectors_doc_type
    ON public.chunk_vectors (doc_type);
CREATE INDEX IF NOT EXISTS idx_chunk_vectors_trust_level
    ON public.chunk_vectors (trust_level);
---------------------------------------------------------------------------

Note: the requirements spec refers to these tables as `legal_chunks` and
`legal_chunk_embeddings`.  The live schema uses `chunks` and `chunk_vectors`
respectively — the mapping is 1-to-1 in purpose.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Date, DateTime, SmallInteger, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# public.chunks — source-of-truth for chunk text and metadata (written by D3)
# ---------------------------------------------------------------------------

class Chunk(Base):
    """
    Maps to public.chunks.

    D4 reads this table to obtain chunk metadata (jurisdiction, doc_type, etc.)
    when upserting into chunk_vectors.  D4 never writes to this table.
    """

    __tablename__ = "chunks"
    __table_args__ = {"schema": "public"}

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True
    )
    doc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    jurisdiction: Mapped[str] = mapped_column(Text, nullable=False)
    doc_type: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(Text, nullable=False, default="en")
    trust_level: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    effective_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    source_org: Mapped[str | None] = mapped_column(Text, nullable=True)


# ---------------------------------------------------------------------------
# public.chunk_vectors — D4 pgvector embedding table
# ---------------------------------------------------------------------------

class ChunkVector(Base):
    """
    Maps to public.chunk_vectors.

    The embedding column is populated (or updated) by POST /d4/embed.
    Cosine similarity search is performed over this table by POST /d4/retrieve.
    """

    __tablename__ = "chunk_vectors"
    __table_args__ = {"schema": "public"}

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True
    )
    doc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    jurisdiction: Mapped[str] = mapped_column(Text, nullable=False)
    doc_type: Mapped[str] = mapped_column(Text, nullable=False)
    trust_level: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=3
    )
    effective_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    source_org: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Vector(768): pgvector column — nullable until POST /d4/embed is called
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(768), nullable=True
    )
    embedding_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, server_default=func.now()
    )
