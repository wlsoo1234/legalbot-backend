"""Administrative source-ingestion API backed by a durable Pub/Sub worker."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.config import get_settings
from app.deps_rag import get_legal_knowledge_repo
from app.rag_auth import require_rag_admin
from app.repositories.legal_base import LegalKnowledgeRepository
from app.services.ingest_service import ingest_file, queue_url_document
from app.services.ingest_validation import IngestValidationError, validate_pdf, validate_public_url
from app.services.pipeline_store import get_document, mark_failed
from app.services.rag_ingest_queue import publish_ingest_job

router = APIRouter(prefix="/api/v1/d3", tags=["D3 Ingestion Pipeline"])
logger = logging.getLogger(__name__)
Admin = Annotated[None, Depends(require_rag_admin)]


def _require_durable_store() -> None:
    if not get_settings().use_alloydb:
        raise HTTPException(
            status_code=503,
            detail="Durable ingestion requires USE_ALLOYDB=true.",
        )


class IngestUrlBody(BaseModel):
    url: str = Field(..., min_length=8, max_length=2048)


class IngestResponse(BaseModel):
    doc_id: str
    chunk_count: int = 0
    page_count: int = 0
    status: str = "pending"


class DocumentStatus(BaseModel):
    doc_id: str
    source_id: str
    status: str
    page_count: int | None = None
    chunk_count: int | None = None
    extracted_text_length: int | None = None
    error_message: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None
    attempt_count: int = 0


async def _read_upload_limited(file: UploadFile, limit: int) -> bytes:
    body = bytearray()
    while part := await file.read(1024 * 1024):
        body.extend(part)
        if len(body) > limit:
            raise IngestValidationError(
                f"The uploaded PDF exceeds the {limit // (1024 * 1024)} MB limit."
            )
    return bytes(body)


@router.get("/documents/{doc_id}", response_model=DocumentStatus)
async def get_document_status(doc_id: str, _admin: Admin) -> DocumentStatus:
    doc = await run_in_threadpool(get_document, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DocumentStatus(
        doc_id=str(doc["doc_id"]),
        source_id=str(doc["source_id"]),
        status=doc["status"],
        page_count=doc.get("page_count"),
        chunk_count=doc.get("chunk_count"),
        extracted_text_length=doc.get("extracted_text_length"),
        error_message=doc.get("error_message"),
        created_at=doc.get("created_at"),
        started_at=doc.get("started_at"),
        updated_at=doc.get("updated_at"),
        completed_at=doc.get("completed_at"),
        attempt_count=doc.get("attempt_count") or 0,
    )


async def _publish_or_fail(doc_id: str, source_id: str) -> None:
    try:
        message_id = await publish_ingest_job(doc_id, source_id)
        logger.info("rag_ingest_published doc_id=%s source_id=%s message_id=%s", doc_id, source_id, message_id)
    except Exception as exc:
        detail = f"Could not publish ingestion job: {type(exc).__name__}: {exc}"
        await run_in_threadpool(mark_failed, doc_id, detail)
        raise HTTPException(status_code=503, detail=detail) from exc


@router.post("/sources/{source_id}/ingest/file", response_model=IngestResponse, status_code=202)
async def ingest_source_file(
    source_id: str,
    _admin: Admin,
    file: UploadFile = File(...),
    repo: LegalKnowledgeRepository = Depends(get_legal_knowledge_repo),
) -> IngestResponse:
    _require_durable_store()
    source = await run_in_threadpool(repo.get_source, source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found.")
    if source.source_type not in {"pdf", "manual_upload"}:
        raise HTTPException(status_code=400, detail="This source must use the URL ingestion endpoint.")

    try:
        max_bytes = get_settings().rag_max_upload_mb * 1024 * 1024
        body = await _read_upload_limited(file, max_bytes)
        validate_pdf(
            file.filename or "",
            file.content_type,
            body,
            max_bytes=max_bytes,
        )
        doc_id = await run_in_threadpool(ingest_file, source_id, file.filename or "upload.pdf", body)
    except IngestValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingest staging failed: {exc}") from exc

    await _publish_or_fail(doc_id, source_id)
    return IngestResponse(doc_id=doc_id)


@router.post("/sources/{source_id}/ingest/url", response_model=IngestResponse, status_code=202)
async def ingest_source_url(
    source_id: str,
    body: IngestUrlBody,
    _admin: Admin,
    repo: LegalKnowledgeRepository = Depends(get_legal_knowledge_repo),
) -> IngestResponse:
    _require_durable_store()
    source = await run_in_threadpool(repo.get_source, source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found.")
    if source.source_type not in {"url", "faq"}:
        raise HTTPException(status_code=400, detail="This source must use the PDF ingestion endpoint.")
    try:
        await run_in_threadpool(validate_public_url, body.url)
        doc_id = await run_in_threadpool(queue_url_document, source_id, body.url)
    except IngestValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingest staging failed: {exc}") from exc

    await _publish_or_fail(doc_id, source_id)
    return IngestResponse(doc_id=doc_id)
