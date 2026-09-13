"""Cloud Run Pub/Sub push worker for extract → chunk → embed ingestion."""

from __future__ import annotations

import base64
import asyncio
import json
import logging
import time
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from app.deps_rag import get_legal_knowledge_repo
from app.config import get_settings
from app.services.chunk_service import ValidationError, chunk_document
from app.services.embed_service import embed_chunks_for_doc, vectors_complete_for_doc
from app.services.extract_service import extract_document
from app.services.ingest_service import capture_url_document
from app.services.ingest_validation import IngestValidationError
from app.services.pipeline_store import get_document, mark_failed, update_document

logger = logging.getLogger(__name__)
app = FastAPI(title="LegalBot RAG Ingestion Worker", version="1.0.0")


class IngestJob(BaseModel):
    doc_id: str
    source_id: str


def process_document(doc_id: str, source_id: str, delivery_attempt: int = 1) -> dict:
    """Idempotently advance a document through every remaining pipeline stage."""
    if not get_settings().use_alloydb:
        raise RuntimeError("The ingestion worker requires USE_ALLOYDB=true.")
    started = time.perf_counter()
    document = get_document(doc_id)
    if document is None:
        raise ValueError(f"Document not found: {doc_id}")
    if str(document["source_id"]) != source_id:
        raise ValueError("Ingestion job source_id does not match the document.")
    if document["status"] == "ready":
        return {"doc_id": doc_id, "status": "ready", "idempotent": True}

    attempt_count = max(int(document.get("attempt_count") or 0) + 1, delivery_attempt)

    repository = get_legal_knowledge_repo()
    source = repository.get_source(source_id)
    if source is None:
        raise ValueError(f"Source not found: {source_id}")

    stage_times: dict[str, int] = {}
    update_document(
        doc_id,
        started_at=document.get("started_at") or datetime.now(timezone.utc),
        error_message=None,
        attempt_count=attempt_count,
    )

    if not document.get("gcs_uri_raw"):
        stage = time.perf_counter()
        capture_url_document(doc_id)
        stage_times["fetch_ms"] = int((time.perf_counter() - stage) * 1000)
        document = get_document(doc_id) or document

    if document["status"] in {"pending", "failed"}:
        stage = time.perf_counter()
        extract_document(doc_id)
        stage_times["extract_ms"] = int((time.perf_counter() - stage) * 1000)
        document = get_document(doc_id) or document

    if document["status"] == "extracted":
        stage = time.perf_counter()
        chunk_document(doc_id, source, repository)
        stage_times["chunk_ms"] = int((time.perf_counter() - stage) * 1000)
        document = get_document(doc_id) or document

    if document["status"] == "chunked":
        stage = time.perf_counter()
        embedded = embed_chunks_for_doc(doc_id)
        if not vectors_complete_for_doc(doc_id):
            raise RuntimeError("Not every chunk has a compatible embedding vector.")
        stage_times["embed_ms"] = int((time.perf_counter() - stage) * 1000)
        update_document(
            doc_id,
            status="ready",
            completed_at=datetime.now(timezone.utc),
            error_message=None,
        )
        repository.link_doc_to_source(source_id, doc_id)
    else:
        embedded = 0

    final = get_document(doc_id) or {}
    stage_times["total_ms"] = int((time.perf_counter() - started) * 1000)
    logger.info(
        "rag_ingest_complete doc_id=%s source_id=%s status=%s chunks=%s embedded=%d timings=%s",
        doc_id,
        source_id,
        final.get("status"),
        final.get("chunk_count"),
        embedded,
        stage_times,
    )
    return {
        "doc_id": doc_id,
        "source_id": source_id,
        "status": final.get("status"),
        "chunk_count": final.get("chunk_count"),
        "embedded": embedded,
        "stage_times_ms": stage_times,
    }


def _decode_pubsub(envelope: dict) -> tuple[IngestJob, int]:
    message = envelope.get("message") or {}
    encoded = message.get("data")
    if not encoded:
        raise ValueError("Pub/Sub message.data is required.")
    try:
        payload = json.loads(base64.b64decode(encoded, validate=True))
    except Exception as exc:
        raise ValueError("Pub/Sub message.data is not valid base64 JSON.") from exc
    attempt = int(envelope.get("deliveryAttempt") or 1)
    return IngestJob.model_validate(payload), attempt


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "service": "rag-ingest-worker"}


@app.post("/tasks/rag-ingest")
async def ingest_push(request: Request) -> dict:
    try:
        job, attempt = _decode_pubsub(await request.json())
    except (ValueError, json.JSONDecodeError) as exc:
        logger.error("rag_ingest_invalid_message error=%s", exc)
        return {"status": "ignored", "error": str(exc)}

    try:
        return await asyncio.to_thread(process_document, job.doc_id, job.source_id, attempt)
    except (IngestValidationError, ValidationError, ValueError) as exc:
        mark_failed(job.doc_id, f"{type(exc).__name__}: {exc}")
        logger.error("rag_ingest_terminal doc_id=%s attempt=%d error=%s", job.doc_id, attempt, exc)
        return {"doc_id": job.doc_id, "status": "failed", "error": str(exc)}
    except Exception as exc:
        update_document(job.doc_id, error_message=f"{type(exc).__name__}: {exc}"[:2000])
        logger.exception("rag_ingest_retry doc_id=%s attempt=%d", job.doc_id, attempt)
        document = get_document(job.doc_id) or {}
        attempts = int(document.get("attempt_count") or attempt)
        if attempts >= 5:
            mark_failed(job.doc_id, f"Embedding or pipeline failure after {attempts} attempts: {exc}")
            return {"doc_id": job.doc_id, "status": "failed", "error": str(exc)}
        raise HTTPException(status_code=503, detail="Transient ingestion failure; retry requested.") from exc
