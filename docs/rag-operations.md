# RAG Operations and Deployment

## Prerequisites

- Python 3.11 or newer is recommended.
- An AlloyDB or PostgreSQL instance with the `vector` extension.
- Vertex AI access for the configured chat and embedding models.
- A Cloud Storage bucket and Pub/Sub topic named `legal-ingest` (or the configured equivalent).
- API and worker service accounts with least-privilege access. Prefer Cloud Run service identity / ADC; do not bake JSON keys into images.

This revision removes the previously tracked service-account JSON file. Because deletion does not remove a secret from Git history, revoke/rotate that credential in Google Cloud and purge it from repository history before sharing the repository externally.

Copy `.env.example` to `.env` for local development. Set `RAG_ADMIN_API_KEY` to a long random value and send it as `X-Admin-API-Key` on all source and ingestion administration requests.

## Schema migration

For a new database:

```bash
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f app/db/schema.sql
```

For an existing LegalBot database:

```bash
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f app/db/migrations/002_complete_rag.sql
```

The migration is idempotent. It adds generated full-text search data, the GIN index, vector model provenance, URL-job fields, timestamps, and hybrid retrieval scores.

After changing `RAG_EMBEDDING_MODEL`, or after applying the migration to vectors with no model metadata, run:

```bash
python -m app.commands.reindex_rag
```

Use `--force` to re-embed the entire corpus.

## Local processes

Run the API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Run the worker as a separate service:

```bash
uvicorn worker.main:app --host 0.0.0.0 --port 8080
```

Create a Pub/Sub push subscription whose endpoint is the worker's `/tasks/rag-ingest` route. On Cloud Run, require authentication and configure Pub/Sub push OIDC with a service account granted `roles/run.invoker` on the worker.

Example Google Cloud setup (replace placeholders):

```bash
gcloud pubsub topics create legal-ingest --project PROJECT_ID

gcloud run deploy legalbot-rag-worker \
  --source . \
  --region REGION \
  --project PROJECT_ID \
  --service-account WORKER_SERVICE_ACCOUNT \
  --no-allow-unauthenticated \
  --command uvicorn \
  --args worker.main:app,--host,0.0.0.0,--port,8080 \
  --set-env-vars USE_ALLOYDB=true,RAG_INGEST_TOPIC=legal-ingest

gcloud pubsub subscriptions create legal-ingest-worker \
  --topic legal-ingest \
  --push-endpoint WORKER_URL/tasks/rag-ingest \
  --push-auth-service-account PUSH_SERVICE_ACCOUNT \
  --dead-letter-topic legal-ingest-dead-letter \
  --max-delivery-attempts 5 \
  --project PROJECT_ID
```

Deploy the API separately with the same database, embedding-model, project, bucket, and topic configuration. Grant its identity Pub/Sub publisher and Storage object-creator permissions. Grant the worker Storage object access, AlloyDB client/database permissions, and Vertex AI user access.

## Health checks

- `GET /` — process liveness.
- `GET /api/v1/rag/healthz` — public non-secret configuration summary.
- `GET /api/v1/rag/readyz` — database, pgvector, migration, model, and worker-topic readiness.
- `GET /api/v1/d3/documents/{doc_id}` — admin-protected ingestion status and failure detail.

Do not use the readiness route as a liveness probe: a dependency outage should remove an instance from serving traffic without restarting it repeatedly.

## API examples

Register a source:

```bash
curl -X POST http://localhost:8000/api/v1/rag/sources \
  -H 'Content-Type: application/json' \
  -H "X-Admin-API-Key: $RAG_ADMIN_API_KEY" \
  -d '{
    "jurisdiction": "MY",
    "doc_type": "act",
    "trust_level": 1,
    "source_type": "pdf",
    "title": "Example Act",
    "source_org": "Government of Malaysia",
    "language": "en"
  }'
```

Queue a PDF and poll it:

```bash
curl -X POST http://localhost:8000/api/v1/d3/sources/SOURCE_ID/ingest/file \
  -H "X-Admin-API-Key: $RAG_ADMIN_API_KEY" \
  -F file=@example.pdf

curl http://localhost:8000/api/v1/d3/documents/DOC_ID \
  -H "X-Admin-API-Key: $RAG_ADMIN_API_KEY"
```

Ask a grounded question:

```bash
curl -X POST http://localhost:8000/api/v1/chat/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What process must a landlord follow?","jurisdiction":"MY","top_k":5}'
```

Send the returned `session_id` on follow-ups. The consultation compatibility surface accepts the same optional filters under `POST /api/v1/assistant/consultation/send` with the question in `text`.

## Failure behavior

- Missing server admin-key configuration returns `503`; a missing or wrong request key returns `401`.
- Invalid PDF/URL inputs are rejected before publishing.
- Broker failure marks the document failed and returns `503`, so callers never receive a false `202`.
- Terminal validation failures, including image-only PDFs with insufficient extracted text, become `failed` with a clear status message. OCR is not included.
- Transient worker failures request redelivery. After five delivery attempts the worker records `failed` and acknowledges the message.
