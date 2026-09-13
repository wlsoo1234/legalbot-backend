# Manual Ingestion Test

Use the current instructions in [RAG Operations and Deployment](rag-operations.md#api-examples), or run [smoke_rag_pipeline.py](smoke_rag_pipeline.py) for an end-to-end check.

All source and D3 operations require `X-Admin-API-Key`. Successful ingestion is asynchronous and follows:

```text
pending → extracted → chunked → ready
                              ↘ failed
```

Only `POST /api/v1/chat/ask` provides the dedicated RAG answer API. Raw `/d4/*` operations and `/api/v1/rag/ask` are not mounted.
