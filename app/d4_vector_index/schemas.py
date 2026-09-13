"""
Pydantic v2 request / response schemas for the D4 vector index API.
"""

from __future__ import annotations

import uuid
from typing import Optional

from pydantic import BaseModel, Field, field_validator

EMBEDDING_DIM: int = 768


# ---------------------------------------------------------------------------
# POST /d4/embed
# ---------------------------------------------------------------------------

class EmbedRequest(BaseModel):
    """Body for storing or updating an embedding."""

    chunk_id: uuid.UUID
    embedding: list[float] = Field(
        ...,
        description=f"Embedding vector of exactly {EMBEDDING_DIM} floats.",
    )
    model: str = Field(
        "text-embedding-004",
        description="Name of the embedding model used.",
    )

    @field_validator("embedding")
    @classmethod
    def validate_dimension(cls, v: list[float]) -> list[float]:
        if len(v) != EMBEDDING_DIM:
            raise ValueError(
                f"embedding must have exactly {EMBEDDING_DIM} dimensions, "
                f"got {len(v)}."
            )
        return v


class EmbedResponse(BaseModel):
    """Confirmation that the embedding was stored."""

    chunk_id: uuid.UUID
    status: str = "stored"


# ---------------------------------------------------------------------------
# POST /d4/retrieve
# ---------------------------------------------------------------------------

class FilterParams(BaseModel):
    """Optional metadata filters applied before cosine ranking."""

    jurisdiction: Optional[str] = None
    doc_type: Optional[str] = None
    language: Optional[str] = None
    trust_max: Optional[int] = Field(
        None,
        ge=1,
        le=5,
        description="Include only chunks whose trust_level <= trust_max.",
    )


class RetrieveRequest(BaseModel):
    """Body for top-K cosine-similarity retrieval.

    Provide either ``query_text`` (auto-embedded server-side) or a
    pre-computed ``query_embedding`` vector.  Exactly one is required.
    """

    query_text: Optional[str] = Field(
        None,
        description="Plain-text query. The server embeds it with text-embedding-004.",
    )
    query_embedding: Optional[list[float]] = Field(
        None,
        description=f"Pre-computed query vector of exactly {EMBEDDING_DIM} floats.",
    )
    top_k: int = Field(8, ge=1, le=100)
    filters: FilterParams = Field(default_factory=FilterParams)

    @field_validator("query_embedding")
    @classmethod
    def validate_dimension(cls, v: Optional[list[float]]) -> Optional[list[float]]:
        if v is not None and len(v) != EMBEDDING_DIM:
            raise ValueError(
                f"query_embedding must have exactly {EMBEDDING_DIM} dimensions, "
                f"got {len(v)}."
            )
        return v

    def model_post_init(self, __context) -> None:  # noqa: ANN001
        if self.query_text is None and self.query_embedding is None:
            raise ValueError("Provide either query_text or query_embedding.")


class RetrieveResult(BaseModel):
    """Single ranked result."""

    chunk_id: uuid.UUID
    score: float = Field(
        description="Cosine similarity score in [0, 1]. Higher = more similar."
    )


class RetrieveResponse(BaseModel):
    """Top-K retrieval response."""

    results: list[RetrieveResult]
