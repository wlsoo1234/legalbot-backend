from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.rag_chatbot.retrieval import HybridCandidate
from app.rag_chatbot.schemas import AnswerJSON, ChatRequest
from app.rag_chatbot import service
from app.rag_chatbot import repository
from app.rag_chatbot.schemas import CitationOut
import uuid


@pytest.mark.asyncio
async def test_supported_answer_uses_only_database_citations(monkeypatch):
    saved = {}

    async def history(*args, **kwargs):
        return []

    async def embed(*args, **kwargs):
        return [0.1] * 768

    async def retrieve(*args, **kwargs):
        return [HybridCandidate("known", 1.0, vector_score=0.8, matched_by=("vector", "lexical"))]

    async def details(*args, **kwargs):
        return {
            "known": {
                "chunk_id": "known",
                "text": "A landlord must follow the statute.",
                "doc_title": "Test Act",
                "source_org": "Government",
                "page_start": 2,
                "page_end": 2,
                "section_title": "Section 1",
                "link": "https://example.test/act",
            }
        }

    async def generate(*args, **kwargs):
        return AnswerJSON(
            answer="The indexed statute requires the prescribed process.",
            cited_chunk_ids=["unknown", "known"],
            confidence="high",
            follow_up_questions=[],
        )

    async def save(*args, **kwargs):
        saved.update(kwargs)

    monkeypatch.setattr(service.repository, "fetch_recent_turns", history)
    monkeypatch.setattr(service, "embed_text", embed)
    monkeypatch.setattr(service, "retrieve_hybrid", retrieve)
    monkeypatch.setattr(service.repository, "fetch_chunk_details", details)
    monkeypatch.setattr(service.gemini_client, "ask", generate)
    monkeypatch.setattr(service.repository, "save_exchange", save)

    response = await service.ask(ChatRequest(question="What is required?"), object())
    assert uuid.UUID(response.session_id)
    assert response.answer
    assert [citation.chunk_id for citation in response.citations] == ["known"]
    assert saved["citations"][0].doc_title == "Test Act"


@pytest.mark.asyncio
async def test_no_context_returns_nullable_answer_and_none_confidence(monkeypatch):
    async def empty(*args, **kwargs):
        return []

    async def embed(*args, **kwargs):
        return [0.1] * 768

    async def save(*args, **kwargs):
        return None

    monkeypatch.setattr(service.repository, "fetch_recent_turns", empty)
    monkeypatch.setattr(service, "embed_text", embed)
    monkeypatch.setattr(service, "retrieve_hybrid", empty)
    monkeypatch.setattr(service.repository, "fetch_chunk_details", empty)
    monkeypatch.setattr(service.repository, "save_exchange", save)

    response = await service.ask(ChatRequest(question="Unknown issue"), object())
    assert response.answer is None
    assert response.confidence == "none"
    assert response.citations == []


@pytest.mark.asyncio
async def test_weak_context_caps_model_confidence_at_low(monkeypatch):
    async def empty_history(*args, **kwargs):
        return []

    async def embed(*args, **kwargs):
        return [0.1] * 768

    async def retrieve(*args, **kwargs):
        return [HybridCandidate("weak", 1.0, vector_score=0.1, matched_by=("vector",))]

    async def details(*args, **kwargs):
        return {"weak": {"chunk_id": "weak", "text": "Direct support", "doc_title": "Act"}}

    async def generate(*args, **kwargs):
        return AnswerJSON(answer="Supported", cited_chunk_ids=["weak"], confidence="high")

    async def save(*args, **kwargs):
        return None

    monkeypatch.setattr(service.repository, "fetch_recent_turns", empty_history)
    monkeypatch.setattr(service, "embed_text", embed)
    monkeypatch.setattr(service, "retrieve_hybrid", retrieve)
    monkeypatch.setattr(service.repository, "fetch_chunk_details", details)
    monkeypatch.setattr(service.gemini_client, "ask", generate)
    monkeypatch.setattr(service.repository, "save_exchange", save)
    response = await service.ask(ChatRequest(question="Question"), object())
    assert response.answer == "Supported"
    assert response.confidence == "low"


@pytest.mark.asyncio
async def test_history_repository_scopes_sessions_and_restores_time_order():
    class MappingRows:
        def __init__(self, rows):
            self._rows = rows

        def all(self):
            return self._rows

    class Result:
        def __init__(self, rows):
            self._rows = rows

        def mappings(self):
            return MappingRows(self._rows)

    class FakeSession:
        def __init__(self):
            self.params = []

        async def execute(self, statement, params):
            self.params.append(params)
            rows = {
                "one": [{"question": "new", "answer": "2"}, {"question": "old", "answer": "1"}],
                "two": [],
            }
            return Result(rows[params["session_id"]])

    session = FakeSession()
    first = await repository.fetch_recent_turns(session, "one", limit=6)
    second = await repository.fetch_recent_turns(session, "two", limit=6)
    assert [turn.question for turn in first] == ["old", "new"]
    assert second == []
    assert session.params == [
        {"session_id": "one", "turn_limit": 6},
        {"session_id": "two", "turn_limit": 6},
    ]


@pytest.mark.asyncio
async def test_consultation_adapter_preserves_fields_and_adds_rag_metadata(monkeypatch):
    from app.services import legal_assistant_service as adapter
    from common.schemas import SendMessageRequest

    created = []

    async def create(text, role, session_id):
        created.append((text, role, session_id))
        return {"id": "message", "text": text, "role": role, "createdAt": 1, "sessionId": session_id}

    async def rag_ask(request, db):
        return SimpleNamespace(
            session_id=request.session_id,
            qa_id="qa",
            answer="Grounded answer",
            citations=[],
            confidence="high",
            follow_up_questions=["Next?"],
            disclaimer="Disclaimer",
            model="model",
            retrieval_k=1,
        )

    monkeypatch.setattr(adapter, "create_consultation_message", create)
    monkeypatch.setattr(adapter, "ask", rag_ask)
    result = await adapter.send_consultation_message(
        SendMessageRequest(text="Question", session_id="session"), object()
    )
    assert result["id"] == "message"
    assert result["text"] == "Grounded answer"
    assert result["role"] == "model"
    assert result["createdAt"] == 1
    assert result["sessionId"] == "session"
    assert result["qaId"] == "qa"
    assert len(created) == 2


@pytest.mark.asyncio
async def test_qa_and_citations_commit_once():
    class FakeSession:
        def __init__(self):
            self.executions = []
            self.commits = 0
            self.rollbacks = 0

        async def execute(self, statement, params):
            self.executions.append((str(statement), params))

        async def commit(self):
            self.commits += 1

        async def rollback(self):
            self.rollbacks += 1

    session = FakeSession()
    await repository.save_exchange(
        session,
        qa_id=uuid.uuid4(),
        session_id="session",
        jurisdiction="MY",
        question="Question",
        answer="Answer",
        confidence="high",
        model="model",
        retrieval_k=1,
        retrieval_filters={},
        citations=[
            CitationOut(
                chunk_id="chunk",
                doc_title="Act",
                retrieval_score=1,
                score=0.8,
            )
        ],
    )
    assert len(session.executions) == 2
    assert session.commits == 1
    assert session.rollbacks == 0
