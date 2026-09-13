"""
services/pipeline_store.py — Thin read/write layer for the `documents` pipeline table.

Used by ingest_service, extract_service, and chunk_service so they all share one
DB access pattern and a common in-memory fallback for dev mode (USE_ALLOYDB=false).

documents row lifecycle
-----------------------
pending  →  extracted  →  chunked  →  ready
                                  ↘  failed
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# In-memory fallback (dev mode, USE_ALLOYDB=false)
# ---------------------------------------------------------------------------
_store: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _use_alloydb() -> bool:
    from app.config import get_settings  # noqa: PLC0415
    return get_settings().use_alloydb


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_document(
    *,
    source_id: str,
    gcs_uri_raw: str | None = None,
    source_url: str | None = None,
    version_label: str | None = None,
    content_sha256: str | None = None,
) -> str:
    """
    Insert a new ``documents`` row with ``status='pending'``.
    Returns the generated ``doc_id``.

    Raises ``ValueError`` with the existing ``doc_id`` embedded in the message
    if a document with the same ``content_sha256`` already exists for this
    ``source_id`` (duplicate-content guard).
    """
    # ── duplicate check ────────────────────────────────────────────────────────────
    if content_sha256:
        existing_id = _find_by_sha256(source_id, content_sha256)
        if existing_id:
            raise ValueError(
                f"Duplicate content: a document with the same SHA-256 already "
                f"exists for source_id={source_id!r} (doc_id={existing_id!r}). "
                "Re-ingest is not allowed; delete the existing document first."
            )
    # ─────────────────────────────────────────────────────────────────────────────
    doc_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    if _use_alloydb():
        from app.db.alloydb import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    """
                    INSERT INTO documents
                      (doc_id, source_id, version_label, gcs_uri_raw,
                       source_url, content_sha256, status, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, 'pending', %s, %s)
                    """,
                    (doc_id, source_id, version_label, gcs_uri_raw, source_url,
                     content_sha256, now, now),
                )
            finally:
                cur.close()
    else:
        _store[doc_id] = {
            "doc_id": doc_id,
            "source_id": source_id,
            "version_label": version_label,
            "gcs_uri_raw": gcs_uri_raw,
            "source_url": source_url,
            "gcs_uri_extracted": None,
            "content_sha256": content_sha256,
            "page_count": None,
            "extracted_text_length": None,
            "chunk_count": None,
            "status": "pending",
            "error_message": None,
            "created_at": now,
            "started_at": None,
            "updated_at": now,
            "completed_at": None,
            "attempt_count": 0,
        }

    return doc_id


def get_document(doc_id: str) -> dict[str, Any] | None:
    """Return the raw ``documents`` row dict, or None."""
    if _use_alloydb():
        from app.db.alloydb import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cur = conn.cursor()
            try:
                cur.execute("SELECT * FROM documents WHERE doc_id = %s", (doc_id,))
                row = cur.fetchone()
                if row is None:
                    return None
                cols = [d[0] for d in cur.description]
                return dict(zip(cols, row))
            finally:
                cur.close()
    return _store.get(doc_id)


def update_document(doc_id: str, **fields: Any) -> None:
    """
    Patch arbitrary columns on a ``documents`` row.
    Callers pass keyword args matching column names, e.g.::

        update_document(doc_id, status='extracted',
                        gcs_uri_extracted='gs://...', page_count=12)
    """
    fields["updated_at"] = datetime.now(timezone.utc)

    if _use_alloydb():
        from app.db.alloydb import get_connection  # noqa: PLC0415
        set_clause = ", ".join(f"{col} = %s" for col in fields)
        values = list(fields.values()) + [doc_id]
        with get_connection() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    f"UPDATE documents SET {set_clause} WHERE doc_id = %s",
                    values,
                )
            finally:
                cur.close()
    else:
        if doc_id in _store:
            _store[doc_id].update(fields)


def mark_failed(doc_id: str, error: str) -> None:
    """Transition a document to 'failed' and record the error message."""
    update_document(
        doc_id,
        status="failed",
        error_message=error[:2000],
        completed_at=datetime.now(timezone.utc),
    )


def find_duplicate(
    source_id: str,
    content_sha256: str,
    *,
    exclude_doc_id: str | None = None,
) -> str | None:
    """Return another active document with identical content, if any."""
    found = _find_by_sha256(source_id, content_sha256)
    return None if found == exclude_doc_id else found


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_by_sha256(source_id: str, content_sha256: str) -> str | None:
    """
    Return the ``doc_id`` of any existing non-failed document with the same
    (source_id, content_sha256), or None if no duplicate exists.
    """
    if _use_alloydb():
        from app.db.alloydb import get_connection  # noqa: PLC0415
        with get_connection() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    """
                    SELECT doc_id FROM documents
                     WHERE source_id = %s
                       AND content_sha256 = %s
                       AND status <> 'failed'
                     LIMIT 1
                    """,
                    (source_id, content_sha256),
                )
                row = cur.fetchone()
            finally:
                cur.close()
        return str(row[0]) if row else None
    else:
        for doc in _store.values():
            if (
                doc["source_id"] == source_id
                and doc["content_sha256"] == content_sha256
                and doc["status"] != "failed"
            ):
                return doc["doc_id"]
        return None
