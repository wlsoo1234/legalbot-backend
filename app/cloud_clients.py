"""Compatibility exports for centralized lazy clients in :mod:`app.config`."""

from app.config import (
    get_firestore_client,
    get_gcs_client,
    get_google_credentials as _credentials,
    get_pubsub_client,
)

__all__ = ["get_firestore_client", "get_gcs_client", "get_pubsub_client"]
