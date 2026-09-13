"""
services/gcs.py — Google Cloud Storage helpers.

All functions use Application Default Credentials (ADC).
For local dev: run  `gcloud auth application-default login`
For Cloud Run / GKE: attach a service account with roles/storage.objectAdmin.

GCS_BUCKET should be set in .env; every function that needs it accepts
bucket_name explicitly so callers can override per-request.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Iterator

def get_gcs_client():
    """
    Return a lazily-initialised GCS client.
    If GCS_SA_KEY is set in config, credentials are loaded explicitly from
    that service-account JSON file.  Otherwise falls back to ADC
    (GOOGLE_APPLICATION_CREDENTIALS env var, workload identity, or
    `gcloud auth application-default login`).
    """
    from app.cloud_clients import get_gcs_client as factory
    return factory()


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload_bytes(
    data: bytes,
    bucket_name: str,
    blob_name: str | None = None,
    content_type: str = "application/octet-stream",
    prefix: str = "uploads",
) -> str:
    """
    Upload raw bytes to GCS.

    Parameters
    ----------
    data         : raw bytes to upload.
    bucket_name  : target GCS bucket.
    blob_name    : explicit object path inside the bucket.
                   If omitted, a UUID-based path is generated.
    content_type : MIME type stored as object metadata.
    prefix       : folder prefix used when auto-generating a blob name.

    Returns
    -------
    GCS URI, e.g. ``gs://my-bucket/uploads/abc-123.pdf``
    """
    client = get_gcs_client()
    if blob_name is None:
        ext = _ext_for_content_type(content_type)
        blob_name = f"{prefix}/{uuid.uuid4()}{ext}"
    blob = client.bucket(bucket_name).blob(blob_name)
    blob.upload_from_string(data, content_type=content_type)
    return f"gs://{bucket_name}/{blob_name}"


def upload_text(
    text: str,
    bucket_name: str,
    blob_name: str | None = None,
    prefix: str = "text",
    encoding: str = "utf-8",
) -> str:
    """
    Upload a plain-text string as a UTF-8 blob.

    Returns the GCS URI of the stored object.
    """
    return upload_bytes(
        data=text.encode(encoding),
        bucket_name=bucket_name,
        blob_name=blob_name,
        content_type="text/plain; charset=utf-8",
        prefix=prefix,
    )


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_bytes(bucket_name: str, blob_name: str) -> bytes:
    """
    Download a GCS object as raw bytes.

    Raises ``google.cloud.exceptions.NotFound`` if the blob does not exist.
    """
    client = get_gcs_client()
    return client.bucket(bucket_name).blob(blob_name).download_as_bytes()


def download_text(
    bucket_name: str,
    blob_name: str,
    encoding: str = "utf-8",
) -> str:
    """Download a GCS object and decode it as a string."""
    return download_bytes(bucket_name, blob_name).decode(encoding)


def blob_exists(bucket_name: str, blob_name: str) -> bool:
    """Return True if the specified blob exists in the bucket."""
    client = get_gcs_client()
    return client.bucket(bucket_name).blob(blob_name).exists()


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

def delete_blob(bucket_name: str, blob_name: str, missing_ok: bool = True) -> None:
    """
    Delete a single GCS object.

    Parameters
    ----------
    missing_ok : if True, silently ignore ``NotFound`` errors.
    """
    from google.api_core.exceptions import NotFound  # noqa: PLC0415

    client = get_gcs_client()
    blob = client.bucket(bucket_name).blob(blob_name)
    try:
        blob.delete()
    except NotFound:
        if not missing_ok:
            raise


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------

def list_blobs(
    bucket_name: str,
    prefix: str = "",
    delimiter: str | None = None,
) -> Iterator[str]:
    """
    Yield blob names under *prefix* in the given bucket.

    Parameters
    ----------
    prefix    : object prefix to filter by, e.g. ``"uploads/doc-id/"``
    delimiter : use ``"/"`` to get only immediate children (folder-like listing).
    """
    client = get_gcs_client()
    blobs = client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)
    for blob in blobs:
        yield blob.name


# ---------------------------------------------------------------------------
# Signed URL (time-limited public download link)
# ---------------------------------------------------------------------------

def generate_signed_url(
    bucket_name: str,
    blob_name: str,
    expiration_minutes: int = 15,
    method: str = "GET",
) -> str:
    """
    Generate a V4 signed URL for temporary access to a private blob.

    Requirements:
    - Credentials must be a service account key (not ADC user credentials).
    - The service account needs ``roles/storage.objectViewer`` or higher.

    Parameters
    ----------
    expiration_minutes : how long the URL is valid (default 15 min).
    method             : HTTP verb exposed by the URL (``"GET"`` or ``"PUT"``).

    Returns
    -------
    HTTPS signed URL string.
    """
    client = get_gcs_client()
    blob = client.bucket(bucket_name).blob(blob_name)
    url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(minutes=expiration_minutes),
        method=method,
    )
    return url


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _ext_for_content_type(content_type: str) -> str:
    """Map a MIME type to a file extension for auto-generated blob names."""
    _map = {
        "application/pdf": ".pdf",
        "text/plain": ".txt",
        "application/json": ".json",
        "image/png": ".png",
        "image/jpeg": ".jpg",
    }
    base = content_type.split(";")[0].strip()
    return _map.get(base, "")
