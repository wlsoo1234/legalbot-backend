# LegalBot Current System Architecture

This diagram reflects the upgraded RAG implementation in the current working tree.

```mermaid
flowchart LR
    Tenant["Tenant / frontend"]
    Admin["Knowledge-base administrator"]

    subgraph API["FastAPI API service"]
        RequestLog["Request-ID middleware\nstructured timing logs"]
        Chat["POST /api/v1/chat/ask\ncanonical answer API"]
        Consult["POST /api/v1/assistant/consultation/send\ncompatibility adapter"]
        History["GET /api/v1/assistant/consultation/history\nsession scoped"]
        SourceAPI["/api/v1/rag/sources\nadmin protected"]
        IngestAPI["/api/v1/d3/.../ingest\nadmin protected"]
        Health["/api/v1/rag/healthz + readyz"]
    end

    subgraph Core["Canonical app/rag_chatbot engine"]
        Rewrite["Standalone-query rewrite\nlast 6 session turns"]
        EmbedQuery["768-d query embedding"]
        Vector["Top 20 pgvector candidates"]
        Lexical["Top 20 simple-FTS candidates"]
        RRF["Reciprocal-rank fusion\nk=60, deduplicate"]
        Context["Bounded untrusted context"]
        Answer["Structured Gemini answer\nIDs only"]
        Citations["Server-built citations +\nconfidence enforcement"]
        Persist["Atomic Q&A + citation commit"]
    end

    subgraph Ingestion["Separately deployed worker/main.py"]
        Fetch["Safe URL fetch or GCS PDF"]
        Extract["PDF / HTML extraction"]
        Chunk["Legal chunking\ndeterministic chunk IDs"]
        EmbedDocs["Compatible vector upserts"]
        Lifecycle["pending → extracted → chunked → ready"]
    end

    AlloyDB[("AlloyDB / PostgreSQL\npgvector + GIN FTS")]
    Vertex["Vertex AI\nGemini + embeddings"]
    PubSub["Pub/Sub legal-ingest"]
    GCS[("Cloud Storage\nraw + extracted documents")]
    Firestore[("Firestore\nconsultation display messages")]

    Tenant --> RequestLog --> Chat
    RequestLog --> Consult
    RequestLog --> History
    Admin --> SourceAPI
    Admin --> IngestAPI
    IngestAPI --> GCS
    IngestAPI --> PubSub
    PubSub --> Ingestion
    Ingestion --> Fetch --> Extract --> Chunk --> EmbedDocs --> Lifecycle
    Fetch --> GCS
    Extract --> GCS
    Chunk --> AlloyDB
    EmbedDocs --> Vertex
    EmbedDocs --> AlloyDB

    Chat --> Rewrite --> EmbedQuery
    Consult --> Rewrite
    Rewrite --> AlloyDB
    Rewrite --> Vertex
    EmbedQuery --> Vertex
    EmbedQuery --> Vector
    AlloyDB --> Vector
    AlloyDB --> Lexical
    Vector --> RRF
    Lexical --> RRF
    RRF --> Context --> Answer --> Citations --> Persist
    Answer --> Vertex
    Persist --> AlloyDB
    Consult --> Firestore
    History --> Firestore
    Health --> AlloyDB
```

## Boundaries

- `app/rag_chatbot` is the only answer engine. The legacy synchronous `RagService` and `/api/v1/rag/ask` were removed.
- `/d4/*` is deliberately not mounted. Vector writes and queries are internal repository operations.
- Citation labels, pages, organizations, and links come from database records; Gemini may return only cited chunk IDs.
- The API publishes ingestion jobs and waits for Pub/Sub broker confirmation before returning `202`. It does not perform extraction in an in-process background thread.
- The worker is retry-safe: chunk IDs are deterministic and vector writes are upserts tagged with the configured embedding model.
- Agreement analysis and agreement generation remain separate product flows using Firestore, Cloud Storage, Pub/Sub, and Gemini.
