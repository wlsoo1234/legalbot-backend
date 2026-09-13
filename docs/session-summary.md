# Session Summary — LegalBot Backend RAG Pipeline (Historical)

> This document describes an earlier implementation. For the active architecture and APIs, see [current-system-architecture.md](current-system-architecture.md) and [rag-operations.md](rag-operations.md).
**Date:** 2026-02-26  
**Repo:** `legalbot-backend` (FastAPI, Python 3.11+)

---

## What was built

A full **3-step document ingestion pipeline** + **RAG query API** for a tenant rights legal chatbot targeting Malaysian law.

### Pipeline steps
```
Step 0  POST /api/v1/rag/sources          → register source metadata → source_id
Step 1  POST /api/v1/d3/sources/{id}/ingest/file  → file → GCS raw blob → documents row (status=pending)
        POST /api/v1/d3/sources/{id}/ingest/url   → fetch URL → GCS snapshot → documents row
Step 2  (auto)  extract_service           → download raw → PDF/HTML extraction → GCS extracted JSON (status=extracted)
Step 3  (auto)  chunk_service             → overlapping chunks + validation → chunks rows (status=ready)
Query   POST /api/v1/rag/ask              → embed question → pgvector search → Gemini → citations
```

---

## Files created

| File | Purpose |
|---|---|
| `app/db/schema.sql` | Full AlloyDB DDL (rewritten to match live pg_dump) |
| `app/schemas_rag.py` | All Pydantic models |
| `app/repositories/legal_base.py` | Abstract interfaces (D3 + D4) |
| `app/repositories/legal_memory.py` | In-memory implementation (dev) |
| `app/repositories/legal_alloydb.py` | AlloyDB implementation (prod) |
| `app/services/embeddings.py` | `DummyEmbeddingProvider` + `GeminiEmbeddingProvider` |
| `app/services/rag_service.py` | `ingest_document()` + `answer_question()` |
| `app/services/gcs.py` | `upload_bytes()`, `download_bytes()`, `upload_text()` |
| `app/services/ingest_service.py` | Step 1 |
| `app/services/extract_service.py` | Step 2 |
| `app/services/chunk_service.py` | Step 3 + validation |
| `app/services/pipeline_store.py` | `documents` table DB layer |
| `app/routers/rag.py` | RAG query + source registry endpoints |
| `app/routers/d3.py` | File/URL ingest endpoints |
| `app/deps_rag.py` | FastAPI dependency singletons |
| `app/db/alloydb.py` | AlloyDB connector |
| `docs/testing-ingest.md` | Full curl/PowerShell test guide |

## Files modified

| File | Change |
|---|---|
| `app/main.py` | Added `rag` and `d3` routers |
| `app/config.py` | Added `google_cloud_project`, `pubsub_topic_ingest`, AlloyDB + embedding toggles |
| `app/firestore.py` | ADC fallback if `serviceAccountKey.json` missing |
| `app/deps_rag.py` | Fixed FastAPI dep signature bug (see Bugs Fixed) |
| `requirements.txt` | Added `numpy`, `httpx`, `beautifulsoup4`, `lxml`, `pymupdf`, `pgvector`, `google-cloud-alloydb-connector[psycopg2]` |

---

## Database schema (AlloyDB, live)

```
legal_sources     source registry
  source_id UUID PK
  jurisdiction TEXT NOT NULL
  doc_type     TEXT NOT NULL
  trust_level  SMALLINT NOT NULL  (1=gov, 2=ngo, 3-5=other)
  source_type  TEXT CHECK IN ('url','pdf','faq','manual_upload')
  status       TEXT CHECK IN ('registered','ingested','failed','disabled')
  effective_date DATE
  language     TEXT DEFAULT 'en'
  tags         TEXT[]
  source_org   TEXT
  title        TEXT
  canonical_url TEXT

documents         pipeline tracking per doc
  doc_id UUID PK
  source_id UUID FK → legal_sources
  status TEXT CHECK IN ('pending','extracted','chunked','ready','failed')
  gcs_uri_raw   TEXT
  gcs_uri_extracted TEXT
  content_sha256 TEXT           ← dedup guard
  page_count INTEGER
  extracted_text_length INTEGER
  version_label TEXT
  error_message TEXT

chunks            text chunks (RAG retrieval units)
  chunk_id UUID PK
  doc_id UUID FK → documents
  chunk_index INTEGER
  text TEXT
  section_title TEXT
  page_start, page_end INTEGER
  char_start, char_end INTEGER
  jurisdiction, doc_type, language TEXT
  trust_level SMALLINT
  effective_date DATE
  source_org TEXT
  tags TEXT[]
  version_label TEXT
  token_estimate INTEGER
  url_fragment TEXT

chunk_vectors     pgvector embeddings (768-dim, Gemini text-embedding-004)
  chunk_id UUID PK FK → chunks
  doc_id UUID
  jurisdiction, doc_type TEXT
  trust_level SMALLINT
  effective_date DATE
  source_org TEXT
  embedding vector(768)
  INDEX: ivfflat cosine (lists=100)

legal_documents   logical doc record used by RAG answer builder
  doc_id UUID PK
  source_id UUID FK
  jurisdiction, doc_type TEXT
  provider, title, url TEXT
```

---

## GCS bucket layout

Bucket: `gs://kitahack/`  
(set in `.env` as `GCS_BUCKET=kitahack`)

```
legal/raw/{source_id}/{filename}           ← uploaded PDFs / files
legal/snapshots/{source_id}/snapshot.html  ← URL fetched content
legal/extracted/{source_id}/{doc_id}.json  ← extracted text JSON
```

---

## Key env vars (`.env`)

```bash
LLM_API_KEY=...                    # Google Gemini API key (aistudio.google.com)
GOOGLE_CLOUD_PROJECT=kitahack-488509
GCS_BUCKET=kitahack

USE_ALLOYDB=false                  # true → AlloyDB, false → in-memory
USE_GEMINI_EMBEDDINGS=false        # true → real 768-dim embeddings (needs LLM_API_KEY)

ALLOYDB_INSTANCE_URI=              # projects/.../instances/...  (only when USE_ALLOYDB=true)
ALLOYDB_DB=legalbot
ALLOYDB_USER=
ALLOYDB_PASSWORD=
ALLOYDB_IP_TYPE=PUBLIC
```

ADC is used for GCS + Firestore. Run:
```bash
gcloud auth application-default login
gcloud config set project kitahack-488509
```

---

## Chunk service tunables (`app/services/chunk_service.py`)

```python
CHUNK_CHARS       = 2048   # ~512 tokens
OVERLAP_CHARS     = 400    # overlap between adjacent chunks
MIN_EXTRACTED_CHARS = 200  # minimum text to pass validation

ALLOWED_LANGUAGES = frozenset({"en", "ms", "zh", "zh-hans", "zh-hant", "ta"})
```

---

## Validation rules (enforced in chunk_service before status → ready)

| Rule | Value |
|---|---|
| `extracted_text_length` | > 200 chars |
| `jurisdiction` | non-empty |
| `doc_type` | non-empty |
| `trust_level` | integer 1–5 |
| `language` | in `{en, ms, zh, zh-hans, zh-hant, ta}` |
| `chunk_count` | ≥ 1 |
| `content_sha256` | no duplicate non-failed doc for same source |

Validation failure → HTTP 422, `documents.status = 'failed'`.

---

## RAG query filters (`POST /api/v1/rag/ask`)

```json
{
  "question": "...",
  "jurisdiction": "MY",         // required — always filters
  "doc_type": "tenancy_act",    // optional
  "trust_level_max": 2,         // only chunks from sources ≤ this level
  "effective_date_min": "2022-01-01",  // prefer latest law
  "top_k": 5
}
```

---

## Citation fields (what the UI gets per answer)

```json
{
  "title": "Residential Tenancy Act 2021",
  "provider": "KPKT",
  "page_start": 4, "page_end": 5,
  "section": "PART III — TENANCY DEPOSITS",
  "effective_date": "2022-01-01",
  "source_org": "KPKT",
  "version_label": null,
  "excerpt": "No landlord shall demand..."
}
```

---

## Bugs fixed this session

### 1. `FastAPIError: Invalid args for response field LegalKnowledgeRepository`
**File:** `app/deps_rag.py`  
`get_rag_service()` had abstract class types as default parameter values. FastAPI inspected them at import time and tried to build Pydantic fields from abstract classes.  
**Fix:** Removed all parameters; function now calls singletons directly.

### 2. `FileNotFoundError: ./serviceAccountKey.json`
**File:** `app/firestore.py`  
Hardcoded key file path crashed startup when file absent.  
**Fix:** Check if file exists; fall back to `firestore.Client()` (ADC).

---

## Run the server

```bash
# activate venv first
python -m uvicorn app.main:app --reload --port 8000
# Swagger UI → http://localhost:8000/docs
```

---

## What is NOT yet done (from original plan)

- Job schemas (`JobStatus`, `JobRecord`, `WorkerPayload`)
- Cloud Tasks enqueueing (`app/services/enqueue.py`)
- `worker/` FastAPI app (background processing)
- `GET /jobs/{job_id}` and `GET /jobs/{job_id}/result` endpoints
- Dockerfiles + `docker-compose.yml`
- Makefile
- Unit tests
- `.env.example`
- Formal `report_store.py` migration to repository pattern
