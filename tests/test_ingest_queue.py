from types import SimpleNamespace

import pytest

from app.services import rag_ingest_queue


@pytest.mark.asyncio
async def test_publish_waits_for_broker_confirmation(monkeypatch):
    calls = []

    async def directly(function, *args, **kwargs):
        return function(*args, **kwargs)

    class Future:
        def result(self, timeout):
            calls.append(("confirmed", timeout))
            return "message-1"

    class Publisher:
        def topic_path(self, project, topic):
            calls.append(("topic", project, topic))
            return f"projects/{project}/topics/{topic}"

        def publish(self, path, data, **attributes):
            calls.append(("publish", path, attributes))
            return Future()

    monkeypatch.setattr(
        rag_ingest_queue,
        "get_settings",
        lambda: SimpleNamespace(effective_gcp_project="project", rag_ingest_topic="legal-ingest"),
    )
    monkeypatch.setattr(rag_ingest_queue, "get_pubsub_client", lambda: Publisher())
    monkeypatch.setattr(rag_ingest_queue.asyncio, "to_thread", directly)
    assert await rag_ingest_queue.publish_ingest_job("doc", "source") == "message-1"
    assert calls[-1] == ("confirmed", 30)
