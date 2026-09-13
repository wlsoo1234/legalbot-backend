"""
repositories/legal_base.py — Abstract interfaces for the legal knowledge base (D3)
and vector index (D4).

These interfaces are the single seam between business logic and storage backends.
Swap the in-memory implementation (legal_memory.py) for an AlloyDB+pgvector or
Vertex AI Vector Search implementation without touching any service code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas_rag import (
    ChunkRecord,
    LegalDocument,
    LegalSourceCreate,
    RegisteredSource,
    SourceRegisterRequest,
    SourceUpdateRequest,
)


# ---------------------------------------------------------------------------
# D3 — Legal Knowledge Base
# ---------------------------------------------------------------------------

class LegalKnowledgeRepository(ABC):
    """
    Manages legal source documents and their text chunks (D3 store).

    A single document maps to 1-N chunks produced by the ingestion pipeline.

    Two-step workflow
    -----------------
    1. register_source()  — store metadata, get back a source_id (no text yet).
    2. upsert_document()  — provide text; chunks + vectors are created and
                            the source record is updated with the resulting doc_id.
    """

    # ------------------------------------------------------------------
    # Source registry (Step 0)
    # ------------------------------------------------------------------

    @abstractmethod
    def register_source(self, request: SourceRegisterRequest) -> RegisteredSource:
        """
        Persist source metadata and return a new RegisteredSource.
        The source starts with status='registered' and is transitioned to 'ingested' after text ingestion.
        """
        ...

    @abstractmethod
    def get_source(self, source_id: str) -> RegisteredSource | None:
        """Return a registered source by ID, or None if not found."""
        ...

    @abstractmethod
    def update_source(
        self,
        source_id: str,
        updates: SourceUpdateRequest,
    ) -> RegisteredSource | None:
        """
        Apply a partial update to a registered source.
        Returns the updated record, or None if the source_id does not exist.
        """
        ...

    @abstractmethod
    def list_sources(
        self,
        jurisdiction: str | None = None,
        doc_type: str | None = None,
        status: str | None = None,
    ) -> list[RegisteredSource]:
        """
        Return all registered sources, optionally filtered.

        Parameters
        ----------
        jurisdiction : filter by exact jurisdiction code.
        doc_type     : filter by document type.
        status       : filter by lifecycle status — 'registered', 'ingested',
                       'failed', or 'disabled'.  Pass None to return all.
        """
        ...

    @abstractmethod
    def link_doc_to_source(self, source_id: str, doc_id: str) -> None:
        """
        After successful ingestion, call this to transition the source status
        to 'ingested'.  Idempotent on repeated calls.
        """
        ...

    # ------------------------------------------------------------------
    # Document + chunk store (Step 1)
    # ------------------------------------------------------------------

    @abstractmethod
    def upsert_document(
        self,
        source: LegalSourceCreate,
        text: str,
    ) -> tuple[LegalDocument, list[ChunkRecord]]:
        """
        Persist a document and its derived chunks.

        The implementation is responsible for chunking the raw text and
        generating metadata (tags, timestamps, IDs).  Returns the stored
        document record and all chunk records created.

        If a document with the same (provider, title, jurisdiction) already
        exists, implementations *may* choose to replace or append chunks.
        """
        ...

    @abstractmethod
    def get_document(self, doc_id: str) -> LegalDocument | None:
        """Return a single document by its ID, or None if not found."""
        ...

    @abstractmethod
    def save_document(self, doc: LegalDocument) -> None:
        """
        Persist a pre-built LegalDocument without running the chunking pipeline.
        Used by the Step-1/2/3 ingest pipeline which builds chunks itself.
        On conflict (same doc_id) the row is left unchanged (idempotent).
        """
        ...

    @abstractmethod
    def upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        """
        Persist a batch of pre-built ChunkRecord objects.
        On duplicate chunk_id the row is left unchanged (ON CONFLICT DO NOTHING).
        """
        ...

    @abstractmethod
    def get_chunks(self, chunk_ids: list[str]) -> list[ChunkRecord]:
        """
        Fetch multiple chunk records by their IDs.
        Returns only the chunks that exist; order mirrors the input list.
        """
        ...


# ---------------------------------------------------------------------------
# D4 — Vector Index
# ---------------------------------------------------------------------------

class VectorIndexRepository(ABC):
    """
    Manages dense embeddings for chunk retrieval (D4 store).

    Each entry in the index is keyed by chunk_id and carries:
    - the embedding vector
    - lightweight metadata for pre-filtering (jurisdiction, doc_type, doc_id)
    """

    @abstractmethod
    def upsert_vectors(
        self,
        items: list[tuple[str, list[float], dict]],
    ) -> None:
        """
        Insert or replace embeddings.

        Parameters
        ----------
        items:
            Each element is a 3-tuple:
            (chunk_id: str, embedding: list[float], metadata: dict)
            Metadata should contain at least: doc_id, jurisdiction, doc_type.
        """
        ...

    @abstractmethod
    def query(
        self,
        embedding: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[tuple[str, float]]:
        """
        Return the top-k most similar chunks.

        Parameters
        ----------
        embedding:
            Query vector of the same dimensionality as stored vectors.
        top_k:
            Maximum number of results to return.
        filters:
            Optional key/value pairs to narrow the search.  Supported keys:
            ``jurisdiction`` (str), ``doc_type`` (str), ``doc_id`` (str).

        Returns
        -------
        list of (chunk_id, cosine_similarity_score) sorted descending by score.
        """
        ...
