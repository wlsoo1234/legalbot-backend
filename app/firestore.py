"""Compatibility proxy for the centrally configured lazy Firestore client."""

from app.cloud_clients import get_firestore_client


class _LazyFirestoreProxy:
    def __getattr__(self, name):
        return getattr(get_firestore_client(), name)


db = _LazyFirestoreProxy()
