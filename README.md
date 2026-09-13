# LegalBot Backend

FastAPI backend for Malaysian tenancy-agreement workflows and a grounded, multi-turn legal-information RAG chatbot.

## RAG architecture

The canonical engine is `app/rag_chatbot`. It uses:

- AlloyDB/PostgreSQL with pgvector for 768-dimensional semantic retrieval;
- PostgreSQL `simple` full-text search with a GIN index;
- reciprocal-rank fusion over the top 20 semantic and top 20 lexical matches;
- Vertex AI for query rewriting, embeddings, and schema-constrained answers;
- server-resolved citation metadata and a server-injected legal disclaimer;
- session-scoped history using the six latest Q&A turns;
- a durable Pub/Sub ingestion worker with deterministic chunk IDs and vector upserts.

See [the current architecture diagram](docs/current-system-architecture.md) and [RAG operations guide](docs/rag-operations.md).

## Active RAG interfaces

| Method and path | Access | Purpose |
|---|---|---|
| `POST /api/v1/chat/ask` | Public | Canonical grounded answer API |
| `POST /api/v1/assistant/consultation/send` | Public | Backward-compatible consultation response backed by the same engine |
| `GET /api/v1/assistant/consultation/history?session_id=...` | Public | Session-scoped display history |
| `POST /api/v1/rag/sources` | Admin key | Register a source |
| `GET /api/v1/rag/sources` | Admin key | List sources |
| `GET /api/v1/rag/sources/{source_id}` | Admin key | Fetch source metadata |
| `PATCH /api/v1/rag/sources/{source_id}` | Admin key | Update source metadata |
| `POST /api/v1/d3/sources/{source_id}/ingest/file` | Admin key | Validate, store, and queue a PDF |
| `POST /api/v1/d3/sources/{source_id}/ingest/url` | Admin key | Validate and queue a public URL |
| `GET /api/v1/d3/documents/{doc_id}` | Admin key | Poll ingestion status |
| `GET /api/v1/rag/healthz` | Public | Liveness/configuration summary |
| `GET /api/v1/rag/readyz` | Public | Dependency and migration readiness |

`/api/v1/rag/ask` has been retired. `/d4/*` is not mounted because raw vector operations are internal.

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Apply the schema or migration:

```bash
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f app/db/schema.sql
# Existing installation:
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f app/db/migrations/002_complete_rag.sql
```

Run the API and worker separately:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
uvicorn worker.main:app --host 0.0.0.0 --port 8080
```

Configure the `legal-ingest` Pub/Sub subscription to push authenticated messages to `POST /tasks/rag-ingest` on the worker. Full Cloud Run instructions and IAM notes are in [docs/rag-operations.md](docs/rag-operations.md).

## Chat request

```json
{
  "question": "What process must a landlord follow before eviction?",
  "session_id": "optional-existing-session",
  "jurisdiction": "MY",
  "doc_type": "act",
  "language": "en",
  "trust_level_max": 2,
  "top_k": 5
}
```

If `session_id` is omitted, the API generates a UUID. The response includes `qa_id`, `session_id`, nullable `answer`, verified citations, confidence, follow-up questions, disclaimer, model, and retrieval count. A model-provided citation ID is discarded unless it belongs to the retrieved database records.

## Ingestion lifecycle

```text
pending → extracted → chunked → ready
                              ↘ failed
```

A document becomes `ready` only when every chunk has a vector tagged with the configured embedding model. Changing that model requires reindexing incompatible vectors:

```bash
python -m app.commands.reindex_rag
```

Image-only PDF OCR is outside the current implementation; such files fail with an extraction/insufficient-text message.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q tests
```

Optional deployed smoke scripts remain under `docs/`; they require a running API and configured Google Cloud/AlloyDB resources and are not part of default unit-test discovery.

## Other product areas

Existing agreement upload, analysis, generation, revision, audit, and agreement-specific chat routes remain mounted. They continue to use Firestore, Cloud Storage, Pub/Sub, and Gemini independently of the legal-corpus RAG engine.
