"""
schemas_rag.py — Pydantic models for the D3/D4 legal knowledge base and 6.0 RAG pipeline.

These models are shared between the API layer (routers/rag.py), the ingestion service,
and the worker pipeline.  Keep them here so both the API and worker can import them
without circular dependencies.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# D3 — Source registration (Step 0 — metadata first)
# ---------------------------------------------------------------------------

class SourceRegisterRequest(BaseModel):
    """
    Register a legal source's metadata *before* its text is available.

    **Quick register** (minimum required fields):
        jurisdiction, doc_type, trust_level, source_type

    All other fields are optional and can be filled in later via
    PATCH /api/v1/rag/sources/{source_id}.
    """
    # Minimum required
    jurisdiction: str = Field(
        ...,
        min_length=2,
        max_length=32,
        description="ISO 3166 region code, e.g. 'MY', 'MY-KL'",
    )
    doc_type: str = Field(
        ...,
        min_length=1,
        max_length=80,
        description="Document category: 'act', 'regulation', 'guideline', 'faq', …",
    )
    trust_level: int = Field(
        3,
        ge=1,
        le=5,
        description="Source trust level: 1=gov (highest), 2=ngo, 3=other",
    )
    source_type: Literal["url", "pdf", "faq", "manual_upload"] = Field(
        "pdf",
        description="Media type: 'url', 'pdf', 'faq', 'manual_upload'",
    )

    # Optional — can be supplied now or patched later
    title: str | None = Field(None, description="Human-readable document title")
    canonical_url: str | None = Field(None, description="Canonical URL of the source")
    source_org: str | None = Field(
        None,
        description="Authoring organisation, e.g. 'KPKT', 'Legal Aid Centre'",
    )
    language: str = Field("en", min_length=2, max_length=35, description="BCP-47 language tag")
    effective_date: date | None = Field(None, description="Date the document took effect")
    tags: list[str] = Field(default_factory=list, description="Free-form tags")


class RegisteredSource(BaseModel):
    """
    A legal source as stored in the D3 registry.
    status transitions: 'registered' → 'ingested' | 'failed' | 'disabled'
    """
    source_id: str = Field(..., description="Unique source ID (UUID)")
    source_type: str
    canonical_url: str | None
    title: str | None
    source_org: str | None
    trust_level: int
    jurisdiction: str
    doc_type: str
    language: str
    effective_date: date | None
    tags: list[str]
    created_at: datetime
    updated_at: datetime
    status: str = Field("registered", description="'registered' | 'ingested' | 'failed' | 'disabled'")


class SourceUpdateRequest(BaseModel):
    """
    Partial update for a registered source via PATCH /api/v1/rag/sources/{source_id}.
    Only provided fields are updated; omitted fields remain unchanged.
    """
    title: str | None = None
    canonical_url: str | None = None
    source_org: str | None = None
    jurisdiction: str | None = Field(None, min_length=2, max_length=32)
    doc_type: str | None = Field(None, min_length=1, max_length=80)
    language: str | None = None
    effective_date: date | None = None
    tags: list[str] | None = None
    source_type: Literal["url", "pdf", "faq", "manual_upload"] | None = None
    trust_level: int | None = Field(None, ge=1, le=5)
    status: Literal["registered", "ingested", "failed", "disabled"] | None = None


# ---------------------------------------------------------------------------
# D3 — Legal Knowledge Base
# ---------------------------------------------------------------------------

class LegalSourceCreate(BaseModel):
    """
    Metadata describing a legal source document *before* it is stored.
    Passed in from the caller when ingesting a new document.
    """
    provider: str = Field(..., description="Author/publisher, e.g. 'KPKT', 'Legal Aid'")
    title: str = Field(..., description="Human-readable document title")
    url: str | None = Field(None, description="Canonical URL for the source, if any")
    jurisdiction: str = Field(
        ...,
        description="ISO 3166 region code or sub-region, e.g. 'MY', 'MY-KL', 'MY-JHR'",
    )
    doc_type: str = Field(
        ...,
        description="Document category, e.g. 'act', 'regulation', 'guideline', 'faq'",
    )
    effective_date: date | None = Field(None, description="Date the document took effect")
    language: str = Field("en", description="BCP-47 language tag of the document text")
    trust_level: int = Field(
        3,
        description="Source trust level: 1=gov (highest), 2=ngo, 3=other",
    )


class LegalDocument(BaseModel):
    """A legal document record stored in the knowledge base (D3)."""
    doc_id: str = Field(..., description="Unique document ID (UUID)")
    source_id: str | None = None
    provider: str | None = None
    title: str | None = None
    url: str | None = None
    jurisdiction: str
    doc_type: str
    created_at: datetime | None = None


class ChunkRecord(BaseModel):
    """
    A single text chunk derived from a document pipeline record.
    Mirrors the `chunks` table in the database.
    """
    chunk_id: str = Field(..., description="Unique chunk ID (UUID)")
    doc_id: str = Field(..., description="Parent document ID (documents.doc_id)")
    chunk_index: int = Field(0, description="Sequential index of this chunk within the document")
    text: str = Field(..., description="Raw text content of this chunk")
    token_estimate: int | None = None
    page_start: int | None = Field(None, description="First page number covered (1-based)")
    page_end: int | None = Field(None, description="Last page number covered (1-based)")
    section_title: str | None = Field(None, description="Section heading this chunk belongs to")
    url_fragment: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    jurisdiction: str = ""
    doc_type: str = ""
    effective_date: date | None = None
    language: str = "en"
    source_org: str | None = None
    trust_level: int = 3
    tags: list[str] = Field(default_factory=list, description="Free-form tags inherited from the source")
    version_label: str | None = Field(None, description="Version label from the pipeline document")
    created_at: datetime
