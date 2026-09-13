"""
repositories/legal_memory.py — In-memory implementations of LegalKnowledgeRepository
and VectorIndexRepository for local development and testing.

Drop-in replacement path:
  AlloyDB + pgvector  →  implement the same ABC in legal_alloydb.py
  Vertex AI Vector Search  →  implement VectorIndexRepository in legal_vertex.py
  Then swap the singletons in deps_rag.py; zero service-layer changes needed.
"""

from __future__ import annotations

import re
import uuid
from collections import OrderedDict
from datetime import datetime, timezone

import numpy as np

from app.repositories.legal_base import LegalKnowledgeRepository, VectorIndexRepository
from app.schemas_rag import (
    ChunkRecord,
    LegalDocument,
    LegalSourceCreate,
    RegisteredSource,
    SourceRegisterRequest,
    SourceUpdateRequest,
)

# ---------------------------------------------------------------------------
# Chunking parameters
# ---------------------------------------------------------------------------
CHUNK_TARGET_WORDS: int = 600   # aim for this many words per chunk
CHUNK_MAX_WORDS: int = 800      # hard upper bound before forcing a split
CHUNK_OVERLAP_WORDS: int = 50   # words from previous chunk prepended to next


def _chunk_text(
    text: str,
    doc_id: str,
    source: LegalSourceCreate,
) -> list[ChunkRecord]:
    """
    Split *text* into overlapping chunks and return ChunkRecord objects.

    Strategy
    --------
    1. Split on blank lines to get paragraphs.
    2. Accumulate paragraphs until the word count reaches CHUNK_TARGET_WORDS.
    3. When the limit is reached (or a paragraph would push past CHUNK_MAX_WORDS),
       flush the current chunk and start a new one seeded with the last
       CHUNK_OVERLAP_WORDS words of the previous chunk.
    """
    # Normalise line endings then split into paragraphs
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text.strip()) if p.strip()]

    if not paragraphs:
        return []

    now = datetime.now(timezone.utc)
    chunks: list[ChunkRecord] = []
    current_paras: list[str] = []
    current_words: int = 0
    overlap_text: str = ""  # carried-over tail from the previous chunk

    def _flush(paras: list[str], overlap: str) -> tuple[ChunkRecord, str]:
        body = "\n\n".join(paras)
        full_text = (overlap + " " + body).strip() if overlap else body
        # Build the tail for the NEXT chunk's overlap
        words = full_text.split()
        next_overlap = " ".join(words[-CHUNK_OVERLAP_WORDS:]) if len(words) > CHUNK_OVERLAP_WORDS else ""
        record = ChunkRecord(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            chunk_index=len(chunks),
            text=full_text,
            section_title=None,
            jurisdiction=source.jurisdiction,
            doc_type=source.doc_type,
            language=source.language,
            source_org=source.provider,
            trust_level=source.trust_level,
            created_at=now,
        )
        return record, next_overlap

    for para in paragraphs:
        para_words = len(para.split())

        # If adding this paragraph keeps us under the max, accumulate
        if current_words + para_words <= CHUNK_MAX_WORDS:
            current_paras.append(para)
            current_words += para_words
            # Flush when we've hit the target
            if current_words >= CHUNK_TARGET_WORDS:
                rec, overlap_text = _flush(current_paras, overlap_text)
                chunks.append(rec)
                current_paras = []
                current_words = 0
        else:
            # Flush what we have first (if non-empty)
            if current_paras:
                rec, overlap_text = _flush(current_paras, overlap_text)
                chunks.append(rec)
                current_paras = []
                current_words = 0
            # Now handle the oversized paragraph itself
            current_paras = [para]
            current_words = para_words
            if current_words >= CHUNK_TARGET_WORDS:
                rec, overlap_text = _flush(current_paras, overlap_text)
                chunks.append(rec)
                current_paras = []
                current_words = 0

    # Flush any remainder
    if current_paras:
        rec, _ = _flush(current_paras, overlap_text)
        chunks.append(rec)

    return chunks


# ---------------------------------------------------------------------------
# In-memory LegalKnowledgeRepository
# ---------------------------------------------------------------------------

class InMemoryLegalKnowledgeRepo(LegalKnowledgeRepository):
    """
    Thread-unsafe in-memory store — suitable for single-process local dev only.

    Both dicts are OrderedDicts so iteration order is insertion order,
    which is convenient for debugging.
    """

    def __init__(self) -> None:
        self._docs: OrderedDict[str, LegalDocument] = OrderedDict()
        self._chunks: OrderedDict[str, ChunkRecord] = OrderedDict()
        # Source registry: stores metadata registered before text ingestion
        self._sources: OrderedDict[str, RegisteredSource] = OrderedDict()

    # ------------------------------------------------------------------
    # Source registry (Step 0)
    # ------------------------------------------------------------------

    def register_source(self, request: SourceRegisterRequest) -> RegisteredSource:
        """Create a new RegisteredSource with status='registered'."""
        now = datetime.now(timezone.utc)
        source = RegisteredSource(
            source_id=str(uuid.uuid4()),
            source_type=request.source_type,
            canonical_url=request.canonical_url,
            title=request.title,
            source_org=request.source_org,
            trust_level=request.trust_level,
            jurisdiction=request.jurisdiction,
            doc_type=request.doc_type,
            language=request.language,
            effective_date=request.effective_date,
            tags=list(request.tags),
            created_at=now,
            updated_at=now,
            status="registered",
        )
        self._sources[source.source_id] = source
        return source

    def get_source(self, source_id: str) -> RegisteredSource | None:
        return self._sources.get(source_id)

    def update_source(
        self,
        source_id: str,
        updates: SourceUpdateRequest,
    ) -> RegisteredSource | None:
        """Apply only the non-None fields from *updates* to the stored source."""
        existing = self._sources.get(source_id)
        if existing is None:
            return None
        # Build a dict of only the fields the caller explicitly provided
        patch = {k: v for k, v in updates.model_dump().items() if v is not None}
        patch["updated_at"] = datetime.now(timezone.utc)
        updated = existing.model_copy(update=patch)
        self._sources[source_id] = updated
        return updated

    def list_sources(
        self,
        jurisdiction: str | None = None,
        doc_type: str | None = None,
        status: str | None = None,
    ) -> list[RegisteredSource]:
        results = list(self._sources.values())
        if jurisdiction is not None:
            results = [s for s in results if s.jurisdiction == jurisdiction]
        if doc_type is not None:
            results = [s for s in results if s.doc_type == doc_type]
        if status is not None:
            results = [s for s in results if s.status == status]
        return results

    def link_doc_to_source(self, source_id: str, doc_id: str) -> None:
        """Transition a source's status to 'ingested'."""
        existing = self._sources.get(source_id)
        if existing is None:
            return
        updated = existing.model_copy(update={
            "status": "ingested",
            "updated_at": datetime.now(timezone.utc),
        })
        self._sources[source_id] = updated

    # ------------------------------------------------------------------
    # Document + chunk store (Step 1)
    # ------------------------------------------------------------------

    def upsert_document(
        self,
        source: LegalSourceCreate,
        text: str,
    ) -> tuple[LegalDocument, list[ChunkRecord]]:
        """
        Store the document + all derived chunks.
        If a document with the same (provider+title+jurisdiction) exists,
        its old chunks are removed and replaced.
        """
        # Check for an existing doc with the same logical identity
        existing_doc_id: str | None = None
        for doc in self._docs.values():
            if (
                doc.provider == source.provider
                and doc.title == source.title
                and doc.jurisdiction == source.jurisdiction
            ):
                existing_doc_id = doc.doc_id
                break

        doc_id = existing_doc_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        # Remove old chunks for this doc if replacing
        if existing_doc_id:
            stale = [cid for cid, c in self._chunks.items() if c.doc_id == doc_id]
            for cid in stale:
                del self._chunks[cid]

        doc = LegalDocument(
            doc_id=doc_id,
            source_id=None,
            provider=source.provider,
            title=source.title,
            url=source.url,
            jurisdiction=source.jurisdiction,
            doc_type=source.doc_type,
            created_at=now,
        )
        self._docs[doc_id] = doc

        new_chunks = _chunk_text(text, doc_id, source)
        for chunk in new_chunks:
            self._chunks[chunk.chunk_id] = chunk

        return doc, new_chunks

    def get_document(self, doc_id: str) -> LegalDocument | None:
        return self._docs.get(doc_id)

    def save_document(self, doc: LegalDocument) -> None:
        """Persist a pre-built LegalDocument (no chunking)."""
        if doc.doc_id not in self._docs:
            self._docs[doc.doc_id] = doc

    def upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        """Add pre-built chunks to the in-memory store (no-op on duplicate)."""
        for chunk in chunks:
            if chunk.chunk_id not in self._chunks:
                self._chunks[chunk.chunk_id] = chunk

    def get_chunks(self, chunk_ids: list[str]) -> list[ChunkRecord]:
        return [self._chunks[cid] for cid in chunk_ids if cid in self._chunks]


# ---------------------------------------------------------------------------
# In-memory VectorIndexRepository  (cosine similarity via numpy)
# ---------------------------------------------------------------------------

class InMemoryVectorIndexRepo(VectorIndexRepository):
    """
    Stores dense embeddings in a numpy matrix for O(n) cosine similarity.

    Each entry carries:
    - chunk_id  (str)
    - L2-normalised embedding vector  (1-D np.ndarray)
    - metadata dict  { doc_id, jurisdiction, doc_type }

    For small corpora (<50 k chunks) this is perfectly adequate.
    For larger corpora, swap to pgvector / Vertex AI Vector Search.
    """

    def __init__(self) -> None:
        # Parallel lists — index i corresponds to the same chunk across all three
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None   # shape (n, dim) float32
        self._meta: list[dict] = []

    # ------------------------------------------------------------------
    # VectorIndexRepository interface
    # ------------------------------------------------------------------

    def upsert_vectors(self, items: list[tuple[str, list[float], dict]]) -> None:
        """
        Insert or update embeddings.
        On duplicate chunk_id, the old entry is replaced in-place.
        """
        # Build a lookup map for existing ids
        id_to_idx = {cid: i for i, cid in enumerate(self._ids)}
        new_vecs: list[np.ndarray] = []
        new_ids: list[str] = []
        new_meta: list[dict] = []

        for chunk_id, embedding, metadata in items:
            vec = np.array(embedding, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0.0:
                vec = vec / norm

            if chunk_id in id_to_idx:
                # Replace existing row in-place
                idx = id_to_idx[chunk_id]
                if self._matrix is not None:
                    self._matrix[idx] = vec
                self._meta[idx] = metadata
            else:
                new_ids.append(chunk_id)
                new_vecs.append(vec)
                new_meta.append(metadata)

        if not new_ids:
            return

        new_block = np.stack(new_vecs, axis=0)  # (k, dim)
        if self._matrix is None:
            self._matrix = new_block
        else:
            self._matrix = np.vstack([self._matrix, new_block])

        self._ids.extend(new_ids)
        self._meta.extend(new_meta)

    def query(
        self,
        embedding: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[tuple[str, float]]:
        """
        Cosine-similarity search with optional metadata pre-filtering.

        Supported filter keys: ``jurisdiction``, ``doc_type``, ``doc_id``.
        Unrecognised keys are silently ignored.
        """
        if self._matrix is None or len(self._ids) == 0:
            return []

        # Determine which indices pass the filter
        if filters:
            trust_max = filters.get("trust_level_max")
            date_min = filters.get("effective_date_min")
            allowed = []
            for i, m in enumerate(self._meta):
                # Equality filters
                if not all(
                    m.get(k) == v
                    for k, v in filters.items()
                    if k in ("jurisdiction", "doc_type", "doc_id")
                ):
                    continue
                # trust_level ≤ trust_level_max
                if trust_max is not None:
                    chunk_tl = m.get("trust_level")
                    if chunk_tl is not None and chunk_tl > trust_max:
                        continue
                # effective_date ≥ effective_date_min (None dates pass through)
                if date_min is not None:
                    chunk_ed = m.get("effective_date")
                    if chunk_ed is not None and chunk_ed < date_min:
                        continue
                allowed.append(i)
        else:
            allowed = list(range(len(self._ids)))

        if not allowed:
            return []

        query_vec = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0.0:
            query_vec = query_vec / norm

        # Slice the matrix to only the allowed rows
        idx_arr = np.array(allowed, dtype=np.int32)
        sub_matrix = self._matrix[idx_arr]          # (m, dim)
        scores = sub_matrix @ query_vec              # (m,) cosine similarities

        # Partial sort — get top_k indices within the sub-matrix
        k = min(top_k, len(allowed))
        top_subidx = np.argpartition(scores, -k)[-k:]
        top_subidx = top_subidx[np.argsort(scores[top_subidx])[::-1]]

        return [
            (self._ids[allowed[si]], float(scores[si]))
            for si in top_subidx
        ]
