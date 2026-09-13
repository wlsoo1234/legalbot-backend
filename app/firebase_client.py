"""Compatibility helpers backed by the shared lazy Google Cloud clients."""

from app.cloud_clients import get_firestore_client, get_gcs_client as get_storage_client
from app.config import get_settings

FIRESTORE_COLLECTION = "agreements"


def get_bucket():
    return get_storage_client().bucket(get_settings().gcs_bucket)


def get_agreements_collection():
    return get_firestore_client().collection(FIRESTORE_COLLECTION)
