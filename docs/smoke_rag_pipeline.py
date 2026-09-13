"""Deployed RAG smoke test with no corpus-size assumptions.

Requires a running API, worker, AlloyDB, Vertex AI, GCS, and Pub/Sub.
"""

from __future__ import annotations

import argparse
import os
import time
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def run(
    *,
    pdf: Path,
    title: str,
    question: str,
    follow_up: str,
    doc_type: str = "act",
) -> None:
    base_url = os.getenv("RAG_SMOKE_BASE_URL", "http://localhost:8000").rstrip("/")
    admin_key = os.getenv("RAG_ADMIN_API_KEY", "")
    if not admin_key:
        raise RuntimeError("RAG_ADMIN_API_KEY must be configured for this smoke test.")
    if not pdf.is_file():
        raise FileNotFoundError(pdf)
    admin_headers = {"X-Admin-API-Key": admin_key}

    with httpx.Client(base_url=base_url, timeout=120) as client:
        assert client.get("/").raise_for_status().json()["status"] == "ok"
        assert client.get("/api/v1/rag/readyz").raise_for_status().json()["status"] == "ready"

        source = client.post(
            "/api/v1/rag/sources",
            headers=admin_headers,
            json={
                "jurisdiction": "MY",
                "doc_type": doc_type,
                "trust_level": 1,
                "source_type": "pdf",
                "title": title,
                "source_org": "Government of Malaysia",
                "language": "en",
                "tags": ["smoke-test"],
            },
        ).raise_for_status().json()
        source_id = source["source_id"]

        with pdf.open("rb") as handle:
            job = client.post(
                f"/api/v1/d3/sources/{source_id}/ingest/file",
                headers=admin_headers,
                files={"file": (pdf.name, handle, "application/pdf")},
            ).raise_for_status().json()
        doc_id = job["doc_id"]

        observed = []
        deadline = time.monotonic() + int(os.getenv("RAG_SMOKE_TIMEOUT_SECONDS", "600"))
        while time.monotonic() < deadline:
            status = client.get(
                f"/api/v1/d3/documents/{doc_id}", headers=admin_headers
            ).raise_for_status().json()
            if not observed or status["status"] != observed[-1]:
                observed.append(status["status"])
                print(f"ingestion: {status['status']} chunks={status.get('chunk_count')}")
            if status["status"] == "ready":
                break
            if status["status"] == "failed":
                raise AssertionError(status.get("error_message"))
            time.sleep(3)
        else:
            raise TimeoutError(f"Document did not become ready; observed={observed}")

        session_id = str(uuid.uuid4())
        first = client.post(
            "/api/v1/chat/ask",
            json={"question": question, "session_id": session_id, "jurisdiction": "MY", "top_k": 5},
        ).raise_for_status().json()
        assert first["session_id"] == session_id and first["qa_id"]
        assert first["disclaimer"] and first["retrieval_k"] > 0
        if first["answer"] is not None:
            assert first["citations"]
            assert all(citation["chunk_id"] and citation["doc_title"] for citation in first["citations"])

        second = client.post(
            "/api/v1/chat/ask",
            json={"question": follow_up, "session_id": session_id, "jurisdiction": "MY", "top_k": 5},
        ).raise_for_status().json()
        assert second["session_id"] == session_id

        isolated_session = str(uuid.uuid4())
        isolated = client.post(
            "/api/v1/chat/ask",
            json={"question": follow_up, "session_id": isolated_session, "jurisdiction": "MY", "top_k": 5},
        ).raise_for_status().json()
        assert isolated["session_id"] != second["session_id"]

        consultation = client.post(
            "/api/v1/assistant/consultation/send",
            json={"text": question, "session_id": session_id, "jurisdiction": "MY", "top_k": 5},
        ).raise_for_status().json()
        for legacy_field in ("id", "text", "role", "createdAt"):
            assert legacy_field in consultation
        assert consultation["sessionId"] == session_id
        assert "citations" in consultation and "confidence" in consultation

    from app.config import get_settings
    from app.db.alloydb import get_connection

    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT COUNT(*), COUNT(*) FILTER (
                WHERE cv.embedding IS NOT NULL AND cv.embedding_model = %s
            )
              FROM chunks c LEFT JOIN chunk_vectors cv USING (chunk_id)
             WHERE c.doc_id = %s
            """,
            (get_settings().rag_embedding_model, doc_id),
        )
        chunk_count, compatible_vectors = cursor.fetchone()
        assert chunk_count > 0 and compatible_vectors == chunk_count
        cursor.execute(
            "SELECT COUNT(*) FROM qa_pairs WHERE qa_id IN (%s, %s)",
            (first["qa_id"], second["qa_id"]),
        )
        assert cursor.fetchone()[0] == 2
        cursor.close()

    print(
        f"PASS source_id={source_id} doc_id={doc_id} chunks={chunk_count} "
        f"lifecycle={observed} session_id={session_id}"
    )


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--doc-type", default="act")
    parser.add_argument("--question", required=True)
    parser.add_argument("--follow-up", required=True)
    args = parser.parse_args()
    run(pdf=args.pdf, title=args.title, doc_type=args.doc_type, question=args.question, follow_up=args.follow_up)


if __name__ == "__main__":
    cli()
