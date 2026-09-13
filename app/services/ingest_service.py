"""
services/ingest_service.py — Step 1: Accept file bytes or a URL, store raw
content to GCS, create a ``documents`` pipeline record, return the doc_id.

Supported source_type values
-----------------------------
pdf / manual_upload : binary file upload  → stored as-is under
                      ``legal/raw/{source_id}/{filename}``
url / faq           : create a durable URL-backed job; the worker safely
                      fetches and stores the HTML/text snapshot

After this step the document status is ``pending``.
Call extract_service.extract_document(doc_id) next.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def ingest_file(
    source_id: str,
    filename: str,
    file_bytes: bytes,
) -> str:
    """
    Upload a binary file (PDF or manual upload) to GCS raw storage and create
    a ``documents`` row.

    Parameters
    ----------
    source_id  : UUID of the parent ``legal_sources`` row.
    filename   : Original filename (used to derive blob name + content-type).
    file_bytes : Raw binary content of the uploaded file.

    Returns
    -------
    doc_id (str)  — newly created documents.doc_id
    """
    from app.config import get_settings  # noqa: PLC0415
    from app.services.gcs import upload_bytes  # noqa: PLC0415
    from app.services.pipeline_store import create_document  # noqa: PLC0415

    s = get_settings()
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "bin"
    content_type = "application/pdf" if ext == "pdf" else "application/octet-stream"
    sha256 = hashlib.sha256(file_bytes).hexdigest()
    safe_name = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    blob_name = f"legal/raw/{source_id}/{sha256[:16]}-{safe_name}"

    logger.info("rag_ingest_upload source_id=%s content_sha256_prefix=%s", source_id, sha256[:12])
    gcs_uri = upload_bytes(file_bytes, s.gcs_bucket, blob_name, content_type)

    doc_id = create_document(source_id=source_id, gcs_uri_raw=gcs_uri, content_sha256=sha256)
    logger.info("[ingest] created doc_id=%s (file)", doc_id)
    return doc_id


def queue_url_document(source_id: str, url: str) -> str:
    """
    Create a pending URL-backed ``documents`` row. Fetching belongs to the
    durable worker so API restarts cannot lose the job.

    Parameters
    ----------
    source_id : UUID of the parent ``legal_sources`` row.
    url       : Public URL to fetch.

    Returns
    -------
    doc_id (str)  — newly created documents.doc_id
    """
    from app.services.pipeline_store import create_document  # noqa: PLC0415
    doc_id = create_document(source_id=source_id, source_url=url)
    logger.info("[ingest] queued URL doc_id=%s source_id=%s", doc_id, source_id)
    return doc_id


def capture_url_document(doc_id: str) -> str:
    """Fetch a queued URL safely and attach the immutable GCS snapshot."""
    from app.config import get_settings  # noqa: PLC0415
    from app.services.gcs import upload_bytes  # noqa: PLC0415
    from app.services.ingest_validation import fetch_public_url  # noqa: PLC0415
    from app.services.pipeline_store import find_duplicate, get_document, update_document  # noqa: PLC0415

    settings = get_settings()
    document = get_document(doc_id)
    if document is None:
        raise ValueError(f"Document not found: {doc_id}")
    if document.get("gcs_uri_raw"):
        return document["gcs_uri_raw"]
    source_url = document.get("source_url")
    if not source_url:
        raise ValueError(f"Document {doc_id} has neither a file nor a source URL.")

    body, content_type, final_url = fetch_public_url(
        source_url,
        timeout_seconds=settings.rag_fetch_timeout_seconds,
        max_bytes=settings.rag_max_fetch_mb * 1024 * 1024,
    )
    digest = hashlib.sha256(body).hexdigest()
    duplicate = find_duplicate(str(document["source_id"]), digest, exclude_doc_id=doc_id)
    if duplicate:
        raise ValueError(f"Duplicate content already exists as doc_id={duplicate}.")
    extension = "txt" if content_type == "text/plain" else "html"
    blob_name = f"legal/snapshots/{document['source_id']}/{digest[:16]}.{extension}"
    gcs_uri = upload_bytes(body, settings.gcs_bucket, blob_name, content_type)
    update_document(
        doc_id,
        gcs_uri_raw=gcs_uri,
        source_url=final_url,
        content_sha256=digest,
        started_at=document.get("started_at") or datetime.now(timezone.utc),
    )
    return gcs_uri
