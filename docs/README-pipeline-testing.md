# RAG Pipeline Testing

The current smoke runner is [smoke_rag_pipeline.py](smoke_rag_pipeline.py). It validates the active APIs, durable lifecycle, compatible vector count, multi-turn sessions, consultation compatibility, citations, and persistence without assuming a fixed number of chunks.

Prerequisites:

1. Apply `app/db/migrations/002_complete_rag.sql`.
2. Run the API and the separate worker.
3. Configure authenticated Pub/Sub push delivery to the worker.
4. Set `RAG_ADMIN_API_KEY` and the Google Cloud/AlloyDB variables in `.env`.

Run a corpus smoke test:

```bash
python docs/smoke_rag_pipeline.py \
  --pdf docs/Contracts-Act-1950.pdf \
  --title "Contracts Act 1950" \
  --question "What makes an agreement enforceable?" \
  --follow-up "Are there exceptions?"
```

Or use one of the document presets:

```bash
python docs/test_sra_pipeline.py
python docs/test_sma_pipeline.py
python docs/test_distress_act_pipeline.py
```

Unit tests do not call deployed services:

```bash
pytest -q tests
```

The retired `/api/v1/rag/ask` and unmounted `/d4/*` endpoints are intentionally not tested.
