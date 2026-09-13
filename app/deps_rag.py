"""
deps_rag.py — FastAPI dependency providers for the RAG module.

All three stores are module-level singletons (created once per process).
To swap to a persistent backend, replace the singleton construction here —
no router code changes required.

The legacy synchronous answer service has been retired. This module now only
provides the source/chunk repository needed by the ingestion API and worker.
"""

from __future__ import annotations

from functools import lru_cache

from app.repositories.legal_base import LegalKnowledgeRepository
from app.repositories.legal_memory import InMemoryLegalKnowledgeRepo


# ---------------------------------------------------------------------------
# Repository singleton, created on first dependency resolution.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_legal_knowledge_repo() -> LegalKnowledgeRepository:
    """
    Returns the knowledge repository (D3 store).

    Controlled by the USE_ALLOYDB env var (default: false):
    - false  → InMemoryLegalKnowledgeRepo  (no external deps, for dev/test)
    - true   → AlloyDBKnowledgeRepository  (requires ALLOYDB_* env vars)
    """
    from app.config import get_settings  # noqa: PLC0415
    if get_settings().use_alloydb:
        from app.repositories.legal_alloydb import AlloyDBKnowledgeRepository  # noqa: PLC0415
        return AlloyDBKnowledgeRepository()
    return InMemoryLegalKnowledgeRepo()
