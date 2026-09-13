"""Durable Pub/Sub dispatch for RAG ingestion jobs."""

from __future__ import annotations

import asyncio
import json

from app.cloud_clients import get_pubsub_client
from app.config import get_settings


async def publish_ingest_job(doc_id: str, source_id: str) -> str:
    settings = get_settings()
    project = settings.effective_gcp_project
    if not project or not settings.rag_ingest_topic:
        raise RuntimeError("RAG Pub/Sub project/topic configuration is missing.")
    client = get_pubsub_client()
    topic_path = client.topic_path(project, settings.rag_ingest_topic)
    payload = json.dumps({"doc_id": doc_id, "source_id": source_id}).encode("utf-8")
    future = client.publish(topic_path, data=payload, doc_id=doc_id, source_id=source_id)
    return await asyncio.to_thread(future.result, timeout=30)
