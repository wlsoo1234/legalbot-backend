"""RAG source administration and health routes.

Answer generation is intentionally exposed only by ``/api/v1/chat/ask``.
"""

from __future__ import annotations

from typing import Annotated
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import text

from app.config import get_settings
from app.db.session import get_async_session
from app.deps_rag import get_legal_knowledge_repo
from app.rag_auth import require_rag_admin
from app.repositories.legal_base import LegalKnowledgeRepository
from app.schemas_rag import RegisteredSource, SourceRegisterRequest, SourceUpdateRequest

router = APIRouter(prefix="/api/v1/rag", tags=["RAG Administration"])
Admin = Annotated[None, Depends(require_rag_admin)]
logger = logging.getLogger(__name__)


@router.post("/sources", response_model=RegisteredSource, status_code=status.HTTP_201_CREATED)
async def register_source(
    request: SourceRegisterRequest,
    _admin: Admin,
    repo: LegalKnowledgeRepository = Depends(get_legal_knowledge_repo),
) -> RegisteredSource:
    return await run_in_threadpool(repo.register_source, request)


@router.get("/sources", response_model=list[RegisteredSource])
async def list_sources(
    _admin: Admin,
    jurisdiction: str | None = Query(None),
    doc_type: str | None = Query(None),
    source_status: str | None = Query(None, alias="status"),
    repo: LegalKnowledgeRepository = Depends(get_legal_knowledge_repo),
) -> list[RegisteredSource]:
    return await run_in_threadpool(
        repo.list_sources,
        jurisdiction,
        doc_type,
        source_status,
    )


@router.get("/sources/{source_id}", response_model=RegisteredSource)
async def get_source(
    source_id: str,
    _admin: Admin,
    repo: LegalKnowledgeRepository = Depends(get_legal_knowledge_repo),
) -> RegisteredSource:
    source = await run_in_threadpool(repo.get_source, source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found.")
    return source


@router.patch("/sources/{source_id}", response_model=RegisteredSource)
async def update_source(
    source_id: str,
    updates: SourceUpdateRequest,
    _admin: Admin,
    repo: LegalKnowledgeRepository = Depends(get_legal_knowledge_repo),
) -> RegisteredSource:
    source = await run_in_threadpool(repo.update_source, source_id, updates)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found.")
    return source


@router.get("/healthz")
async def healthz() -> dict:
    settings = get_settings()
    database_configured = (
        bool(settings.alloydb_instance_uri and settings.alloydb_db and settings.alloydb_user)
        if settings.use_alloydb
        else bool(settings.async_database_url)
    )
    return {
        "status": "ok",
        "enabled": settings.rag_enabled,
        "backend": "alloydb" if settings.use_alloydb else "direct-postgres",
        "database_configured": database_configured,
        "chat_model_configured": bool(settings.rag_chat_model),
        "embedding_model_configured": bool(settings.rag_embedding_model),
        "ingest_topic_configured": bool(settings.rag_ingest_topic),
        "admin_key_configured": bool(settings.rag_admin_api_key),
    }


@router.get("/readyz")
async def readyz() -> dict:
    settings = get_settings()
    try:
        async for session in get_async_session():
            result = await session.execute(
                text(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM pg_extension WHERE extname = 'vector'
                    ) AS vector_enabled,
                    EXISTS (
                        SELECT 1
                          FROM information_schema.columns
                         WHERE table_schema = 'public'
                           AND table_name = 'chunks'
                           AND column_name = 'search_vector'
                    ) AS hybrid_schema_ready,
                    NOT EXISTS (
                        SELECT 1
                          FROM public.chunks c
                          JOIN public.documents d ON d.doc_id = c.doc_id
                          LEFT JOIN public.chunk_vectors cv ON cv.chunk_id = c.chunk_id
                         WHERE d.status = 'ready'
                           AND (cv.embedding IS NULL OR cv.embedding_model IS DISTINCT FROM :embedding_model)
                    ) AS vector_space_compatible
                    """
                ),
                {"embedding_model": settings.rag_embedding_model},
            )
            row = result.mappings().one()
            break
    except Exception as exc:
        logger.exception("rag_readiness_database_failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG database is unavailable.",
        ) from exc
    checks = {
        "database": True,
        "pgvector": bool(row["vector_enabled"]),
        "hybrid_schema": bool(row["hybrid_schema_ready"]),
        "vector_space": bool(row["vector_space_compatible"]),
        "chat_model": bool(settings.rag_chat_model),
        "embedding_model": bool(settings.rag_embedding_model)
        and settings.rag_embedding_dimension == 768,
        "worker_topic": bool(settings.rag_ingest_topic)
        and bool(settings.effective_gcp_project),
    }
    if not all(checks.values()):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not-ready", "checks": checks},
        )
    return {"status": "ready", "checks": checks}
