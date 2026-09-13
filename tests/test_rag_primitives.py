from __future__ import annotations

import asyncio
import socket

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.config import get_settings
from app.rag_auth import require_rag_admin
from app.rag_chatbot.prompts import build_context_block
from app.rag_chatbot.retrieval import HybridCandidate, _filters, reciprocal_rank_fusion
from app.rag_chatbot.schemas import ChatRequest
from app.rag_chatbot.service import _is_low_context, _server_citations, _standalone_query
from app.services.chunk_service import deterministic_chunk_id
from app.services.ingest_validation import IngestValidationError, validate_pdf, validate_public_url


def test_chat_request_defaults_and_bounds():
    request = ChatRequest(question="  Can my landlord evict me?  ")
    assert request.question == "Can my landlord evict me?"
    assert request.jurisdiction == "MY"
    assert request.top_k == 5
    with pytest.raises(ValidationError):
        ChatRequest(question="x", top_k=21)


def test_admin_key_missing_invalid_and_valid(monkeypatch):
    monkeypatch.setenv("RAG_ADMIN_API_KEY", "")
    get_settings.cache_clear()
    with pytest.raises(HTTPException) as missing:
        asyncio.run(require_rag_admin("anything"))
    assert missing.value.status_code == 503

    monkeypatch.setenv("RAG_ADMIN_API_KEY", "secret")
    get_settings.cache_clear()
    with pytest.raises(HTTPException) as invalid:
        asyncio.run(require_rag_admin("wrong"))
    assert invalid.value.status_code == 401
    assert asyncio.run(require_rag_admin("secret")) is None
    get_settings.cache_clear()


def test_pdf_validation_checks_extension_mime_magic_and_size():
    validate_pdf("act.pdf", "application/pdf", b"%PDF-1.7\nbody", max_bytes=100)
    with pytest.raises(IngestValidationError):
        validate_pdf("act.txt", "application/pdf", b"%PDF-", max_bytes=100)
    with pytest.raises(IngestValidationError):
        validate_pdf("act.pdf", "text/plain", b"%PDF-", max_bytes=100)
    with pytest.raises(IngestValidationError):
        validate_pdf("act.pdf", "application/pdf", b"not-pdf", max_bytes=100)
    with pytest.raises(IngestValidationError):
        validate_pdf("act.pdf", "application/pdf", b"%PDF-" + b"x" * 100, max_bytes=100)


def _resolver_for(address: str):
    def resolve(host, port, type=socket.SOCK_STREAM):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]
    return resolve


def test_url_validation_blocks_non_http_and_private_addresses():
    assert validate_public_url("https://example.test/law", resolver=_resolver_for("8.8.8.8"))
    with pytest.raises(IngestValidationError):
        validate_public_url("file:///etc/passwd", resolver=_resolver_for("8.8.8.8"))
    with pytest.raises(IngestValidationError):
        validate_public_url("http://internal.test", resolver=_resolver_for("127.0.0.1"))
    with pytest.raises(IngestValidationError):
        validate_public_url("http://metadata.test", resolver=_resolver_for("169.254.169.254"))


def test_deterministic_chunk_ids_are_stable_and_content_sensitive():
    first = deterministic_chunk_id("doc", 0, "same")
    assert first == deterministic_chunk_id("doc", 0, "same")
    assert first != deterministic_chunk_id("doc", 1, "same")
    assert first != deterministic_chunk_id("doc", 0, "changed")


def test_rrf_deduplicates_normalizes_and_is_stable():
    vector = [
        {"chunk_id": "a", "vector_score": 0.9, "lexical_score": None},
        {"chunk_id": "b", "vector_score": 0.8, "lexical_score": None},
        {"chunk_id": "a", "vector_score": 0.7, "lexical_score": None},
    ]
    lexical = [
        {"chunk_id": "b", "vector_score": None, "lexical_score": 0.7},
        {"chunk_id": "c", "vector_score": None, "lexical_score": 0.6},
    ]
    result = reciprocal_rank_fusion(vector, lexical, rrf_k=60, limit=3)
    assert [item.chunk_id for item in result] == ["b", "a", "c"]
    assert result[0].retrieval_score == 1.0
    assert result[0].matched_by == ("vector", "lexical")


def test_retrieval_filters_cover_all_supported_metadata():
    sql, params = _filters("MY", "act", "en", 2)
    assert "c.jurisdiction" in sql
    assert "c.doc_type" in sql
    assert "c.language" in sql
    assert "c.trust_level" in sql
    assert params == {
        "jurisdiction": "MY",
        "doc_type": "act",
        "language": "en",
        "trust_level_max": 2,
    }


def test_context_limit_is_never_exceeded_and_marks_untrusted_text():
    context = build_context_block(
        [{"chunk_id": "a", "text": "x" * 10_000, "doc_title": "Act"}],
        max_chars=500,
    )
    assert len(context) <= 500
    assert "<UNTRUSTED_LEGAL_TEXT>" in context


def test_unknown_model_citations_are_dropped_and_scores_are_preserved():
    candidates = [HybridCandidate("known", 1.0, vector_score=0.81234, matched_by=("vector",))]
    details = {"known": {"chunk_id": "known", "doc_title": "Act", "text": "law"}}
    citations = _server_citations(["unknown", "known", "known"], candidates, details)
    assert [citation.chunk_id for citation in citations] == ["known"]
    assert citations[0].score == 0.81234
    assert citations[0].retrieval_score == 1.0


def test_weak_vector_only_context_is_low_confidence():
    assert _is_low_context(
        [HybridCandidate("a", 1.0, vector_score=0.2, matched_by=("vector",))],
        0.45,
    )
    assert not _is_low_context(
        [HybridCandidate("a", 1.0, lexical_score=0.1, matched_by=("lexical",))],
        0.45,
    )


@pytest.mark.asyncio
async def test_query_rewrite_falls_back_to_original(monkeypatch):
    async def fail(*args, **kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr("app.rag_chatbot.service.gemini_client.rewrite_query", fail)
    request = ChatRequest(question="What about that?", session_id="session")
    assert await _standalone_query(request, [object()]) == request.question
