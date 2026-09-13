from __future__ import annotations

from types import SimpleNamespace

import pytest

from worker import main


@pytest.mark.parametrize(
    ("initial_status", "expected"),
    [
        ("pending", ["extract", "chunk", "embed"]),
        ("extracted", ["chunk", "embed"]),
        ("chunked", ["embed"]),
        ("ready", []),
    ],
)
def test_worker_retries_resume_without_repeating_finished_stages(monkeypatch, initial_status, expected):
    monkeypatch.setattr(main, "get_settings", lambda: SimpleNamespace(use_alloydb=True))
    state = {
        "doc_id": "doc",
        "source_id": "source",
        "status": initial_status,
        "gcs_uri_raw": "gs://bucket/raw.pdf",
        "started_at": None,
        "chunk_count": 1,
    }
    calls = []

    monkeypatch.setattr(main, "get_document", lambda doc_id: dict(state))
    monkeypatch.setattr(main, "update_document", lambda doc_id, **fields: state.update(fields))
    monkeypatch.setattr(
        main,
        "get_legal_knowledge_repo",
        lambda: SimpleNamespace(
            get_source=lambda source_id: SimpleNamespace(source_id=source_id),
            link_doc_to_source=lambda source_id, doc_id: calls.append("link"),
        ),
    )

    def extract(doc_id):
        calls.append("extract")
        state["status"] = "extracted"

    def chunk(doc_id, source, repository):
        calls.append("chunk")
        state["status"] = "chunked"

    def embed(doc_id):
        calls.append("embed")
        return 1

    monkeypatch.setattr(main, "extract_document", extract)
    monkeypatch.setattr(main, "chunk_document", chunk)
    monkeypatch.setattr(main, "embed_chunks_for_doc", embed)
    monkeypatch.setattr(main, "vectors_complete_for_doc", lambda doc_id: True)

    result = main.process_document("doc", "source")
    assert calls[: len(expected)] == expected
    if initial_status == "ready":
        assert result["idempotent"] is True
    else:
        assert result["status"] == "ready"


def test_worker_fetches_url_only_when_raw_object_is_missing(monkeypatch):
    monkeypatch.setattr(main, "get_settings", lambda: SimpleNamespace(use_alloydb=True))
    state = {
        "doc_id": "doc",
        "source_id": "source",
        "status": "pending",
        "gcs_uri_raw": None,
        "started_at": None,
        "chunk_count": 1,
    }
    calls = []
    monkeypatch.setattr(main, "get_document", lambda doc_id: dict(state))
    monkeypatch.setattr(main, "update_document", lambda doc_id, **fields: state.update(fields))
    monkeypatch.setattr(main, "capture_url_document", lambda doc_id: (calls.append("fetch"), state.update(gcs_uri_raw="gs://x")))
    monkeypatch.setattr(main, "extract_document", lambda doc_id: state.update(status="extracted"))
    monkeypatch.setattr(main, "chunk_document", lambda *args: state.update(status="chunked"))
    monkeypatch.setattr(main, "embed_chunks_for_doc", lambda doc_id: 1)
    monkeypatch.setattr(main, "vectors_complete_for_doc", lambda doc_id: True)
    monkeypatch.setattr(
        main,
        "get_legal_knowledge_repo",
        lambda: SimpleNamespace(
            get_source=lambda source_id: SimpleNamespace(source_id=source_id),
            link_doc_to_source=lambda source_id, doc_id: None,
        ),
    )
    main.process_document("doc", "source")
    assert calls == ["fetch"]
