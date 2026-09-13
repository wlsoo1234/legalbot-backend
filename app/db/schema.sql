-- =============================================================================
-- schema.sql — AlloyDB schema for the LegalBot RAG pipeline
--
-- Matches the live database dump (pg_dump 17.5).
-- Run once against your AlloyDB instance:
--   psql "host=<IP> port=5432 dbname=legalbot user=<USER>" -f schema.sql
-- Or via Cloud Shell / AlloyDB Studio.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA public;
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;

-- =============================================================================
-- D3 — Legal Knowledge Base
-- =============================================================================

-- Step 0: source registry (metadata-first registration)
--   trust_level  SMALLINT: 1=gov (highest credibility), 2=ngo, 3=other
--   status       TEXT: 'registered' → 'ingested' | 'failed' | 'disabled'
--   source_type  TEXT: 'url' | 'pdf' | 'faq' | 'manual_upload'
CREATE TABLE IF NOT EXISTS public.legal_sources (
    source_id       UUID        PRIMARY KEY DEFAULT public.uuid_generate_v4(),
    source_type     TEXT        NOT NULL,
    canonical_url   TEXT,
    title           TEXT,
    source_org      TEXT,
    trust_level     SMALLINT    NOT NULL DEFAULT 3,
    jurisdiction    TEXT        NOT NULL,
    doc_type        TEXT        NOT NULL,
    language        TEXT        NOT NULL DEFAULT 'en',
    effective_date  DATE,
    tags            TEXT[]      NOT NULL DEFAULT '{}',
    status          TEXT        NOT NULL DEFAULT 'registered',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT legal_sources_source_type_check
        CHECK (source_type = ANY (ARRAY['url','pdf','faq','manual_upload'])),
    CONSTRAINT legal_sources_status_check
        CHECK (status = ANY (ARRAY['registered','ingested','failed','disabled']))
);

CREATE INDEX IF NOT EXISTS idx_legal_sources_jurisdiction ON public.legal_sources (jurisdiction);
CREATE INDEX IF NOT EXISTS idx_legal_sources_doc_type     ON public.legal_sources (doc_type);
CREATE INDEX IF NOT EXISTS idx_legal_sources_status       ON public.legal_sources (status);


-- Step 1a: document pipeline tracking record (linked to a legal_source)
--   Tracks GCS URIs, extraction status, and processing metadata.
--   status: 'pending' → 'extracted' → 'chunked' → 'ready' | 'failed'
CREATE TABLE IF NOT EXISTS public.documents (
    doc_id                  UUID        PRIMARY KEY DEFAULT public.uuid_generate_v4(),
    source_id               UUID        NOT NULL REFERENCES public.legal_sources (source_id),
    version_label           TEXT,
    gcs_uri_raw             TEXT,
    source_url              TEXT,
    gcs_uri_extracted       TEXT,
    content_sha256          TEXT,
    page_count              INTEGER,
    extracted_text_length   INTEGER,
    status                  TEXT        NOT NULL DEFAULT 'pending',
    error_message           TEXT,
    chunk_count             INTEGER,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at              TIMESTAMPTZ,
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at            TIMESTAMPTZ,
    attempt_count           INTEGER     NOT NULL DEFAULT 0,
    CONSTRAINT documents_status_check
        CHECK (status = ANY (ARRAY['pending','extracted','chunked','ready','failed']))
);

CREATE INDEX IF NOT EXISTS idx_documents_status ON public.documents (status);
CREATE INDEX IF NOT EXISTS idx_documents_updated_at ON public.documents (updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_source_id ON public.documents (source_id);
CREATE INDEX IF NOT EXISTS idx_documents_content_sha256 ON public.documents (source_id, content_sha256);


-- Step 1b: logical document record used by the RAG pipeline
--   Linked to a legal_source; holds enough metadata to build citations.
CREATE TABLE IF NOT EXISTS public.legal_documents (
    doc_id          UUID        PRIMARY KEY,
    source_id       UUID        REFERENCES public.legal_sources (source_id),
    jurisdiction    TEXT        NOT NULL,
    doc_type        TEXT        NOT NULL,
    provider        TEXT,
    title           TEXT,
    url             TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_legal_documents_jurisdiction ON public.legal_documents (jurisdiction);


-- Step 2: text chunks produced from a document pipeline record
--   trust_level  SMALLINT: inherits from the parent legal_source
--   chunk_index  unique sequential position within the parent document
CREATE TABLE IF NOT EXISTS public.chunks (
    chunk_id        UUID        PRIMARY KEY DEFAULT public.uuid_generate_v4(),
    doc_id          UUID        NOT NULL REFERENCES public.documents (doc_id) ON DELETE CASCADE,
    chunk_index     INTEGER     NOT NULL,
    text            TEXT        NOT NULL,
    token_estimate  INTEGER,
    page_start      INTEGER,
    page_end        INTEGER,
    section_title   TEXT,
    url_fragment    TEXT,
    char_start      INTEGER,
    char_end        INTEGER,
    jurisdiction    TEXT        NOT NULL,
    doc_type        TEXT        NOT NULL,
    effective_date  DATE,
    language        TEXT        NOT NULL,
    source_org      TEXT,
    trust_level     SMALLINT    NOT NULL,
    tags            TEXT[]      NOT NULL DEFAULT '{}',  -- added
    version_label   TEXT,                               -- added
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    search_vector   TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('simple'::regconfig,
            COALESCE(section_title, '') || ' ' || COALESCE(text, ''))
    ) STORED,
    UNIQUE (doc_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id        ON public.chunks (doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_jurisdiction  ON public.chunks (jurisdiction);
CREATE INDEX IF NOT EXISTS idx_chunks_trust_level   ON public.chunks (trust_level);
CREATE INDEX IF NOT EXISTS idx_chunks_effective_date ON public.chunks (effective_date);
CREATE INDEX IF NOT EXISTS idx_chunks_retrieval_filters
    ON public.chunks (jurisdiction, doc_type, language, trust_level);
CREATE INDEX IF NOT EXISTS idx_chunks_search_vector
    ON public.chunks USING GIN (search_vector);


-- =============================================================================
-- D4 — Vector Index (pgvector)
-- Embedding dimension: 768 (configured Vertex AI embedding model)
-- =============================================================================

CREATE TABLE IF NOT EXISTS public.chunk_vectors (
    chunk_id        UUID        PRIMARY KEY REFERENCES public.chunks (chunk_id),
    doc_id          UUID        NOT NULL,
    jurisdiction    TEXT        NOT NULL,
    doc_type        TEXT        NOT NULL,
    trust_level     SMALLINT    NOT NULL DEFAULT 3,  -- added
    effective_date  DATE,                            -- added
    source_org      TEXT,                            -- added
    embedding       public.vector(768),  -- nullable: populated after embedding step
    embedding_model TEXT,
    embedded_at     TIMESTAMPTZ
);

-- HNSW approximate nearest-neighbour index (cosine distance).
-- Unlike ivfflat, hnsw updates its graph on every insert — no rebuild needed.
-- m=16 (neighbours per layer), ef_construction=64 (build accuracy).
-- Migrated from ivfflat (lists=100) which required manual rebuilds as data grew.
CREATE INDEX IF NOT EXISTS chunk_vectors_embedding_hnsw_idx
    ON public.chunk_vectors USING hnsw (embedding public.vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_chunk_vectors_jurisdiction  ON public.chunk_vectors (jurisdiction);
CREATE INDEX IF NOT EXISTS idx_chunk_vectors_doc_type      ON public.chunk_vectors (doc_type);
CREATE INDEX IF NOT EXISTS idx_chunk_vectors_trust_level   ON public.chunk_vectors (trust_level);
CREATE INDEX IF NOT EXISTS idx_chunk_vectors_effective_date ON public.chunk_vectors (effective_date);
CREATE INDEX IF NOT EXISTS idx_chunk_vectors_embedding_model ON public.chunk_vectors (embedding_model);


-- =============================================================================
-- Q&A Persistence (RAG Chatbot)
-- =============================================================================

-- One row per /api/v1/chat/ask call
CREATE TABLE IF NOT EXISTS public.qa_pairs (
    qa_id               UUID        PRIMARY KEY DEFAULT public.uuid_generate_v4(),
    session_id          TEXT,                            -- optional caller-supplied conversation ID
    jurisdiction        TEXT        NOT NULL,
    question            TEXT        NOT NULL,
    answer              TEXT,                            -- NULL when context was insufficient
    confidence          TEXT        CHECK (confidence IN ('high', 'low', 'none')),
    model               TEXT        NOT NULL,
    retrieval_k         INTEGER     NOT NULL,
    retrieval_filters   JSONB,                           -- JSON bag: doc_type, trust_level_max, etc.
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_qa_pairs_session_id    ON public.qa_pairs (session_id);
CREATE INDEX IF NOT EXISTS idx_qa_pairs_jurisdiction  ON public.qa_pairs (jurisdiction);
CREATE INDEX IF NOT EXISTS idx_qa_pairs_created_at    ON public.qa_pairs (created_at DESC);


-- One row per cited chunk in a QAPair answer
CREATE TABLE IF NOT EXISTS public.qa_citations (
    cit_id          UUID        PRIMARY KEY DEFAULT public.uuid_generate_v4(),
    qa_id           UUID        NOT NULL REFERENCES public.qa_pairs (qa_id) ON DELETE CASCADE,
    chunk_id        TEXT,                                -- UUID string of the cited chunk
    doc_title       TEXT,
    source_org      TEXT,
    page_start      INTEGER,
    page_end        INTEGER,
    section_title   TEXT,
    link            TEXT,
    score           FLOAT,                               -- cosine similarity (0–1)
    retrieval_score FLOAT,                               -- normalized hybrid rank (0–1)
    confidence      TEXT        CHECK (confidence IN ('high', 'low')),  -- high=verified, low=hallucinated
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_qa_citations_qa_id     ON public.qa_citations (qa_id);
CREATE INDEX IF NOT EXISTS idx_qa_citations_chunk_id  ON public.qa_citations (chunk_id);


-- =============================================================================
-- Helper: auto-update updated_at on legal_sources
-- =============================================================================

CREATE OR REPLACE FUNCTION public.set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_legal_sources_updated_at ON public.legal_sources;
CREATE TRIGGER trg_legal_sources_updated_at
    BEFORE UPDATE ON public.legal_sources
    FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

DROP TRIGGER IF EXISTS trg_documents_updated_at ON public.documents;
CREATE TRIGGER trg_documents_updated_at
    BEFORE UPDATE ON public.documents
    FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();
