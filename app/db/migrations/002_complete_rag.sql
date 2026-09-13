-- Complete RAG: hybrid search, embedding provenance, and durable ingest fields.
-- Idempotent and safe to apply after schema.sql from earlier revisions.

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;

ALTER TABLE public.documents
    ADD COLUMN IF NOT EXISTS source_url TEXT,
    ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ADD COLUMN IF NOT EXISTS completed_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS attempt_count INTEGER NOT NULL DEFAULT 0;

ALTER TABLE public.chunks
    ADD COLUMN IF NOT EXISTS search_vector TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('simple'::regconfig,
            COALESCE(section_title, '') || ' ' || COALESCE(text, ''))
    ) STORED;

CREATE INDEX IF NOT EXISTS idx_chunks_search_vector
    ON public.chunks USING GIN (search_vector);

ALTER TABLE public.chunk_vectors
    ADD COLUMN IF NOT EXISTS embedding_model TEXT,
    ADD COLUMN IF NOT EXISTS embedded_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_chunk_vectors_embedding_model
    ON public.chunk_vectors (embedding_model);

ALTER TABLE public.qa_citations
    ADD COLUMN IF NOT EXISTS retrieval_score FLOAT;

CREATE INDEX IF NOT EXISTS idx_documents_status ON public.documents (status);
CREATE INDEX IF NOT EXISTS idx_documents_updated_at ON public.documents (updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_source_id ON public.documents (source_id);
CREATE INDEX IF NOT EXISTS idx_documents_content_sha256
    ON public.documents (source_id, content_sha256);
CREATE INDEX IF NOT EXISTS idx_chunks_retrieval_filters
    ON public.chunks (jurisdiction, doc_type, language, trust_level);

CREATE OR REPLACE FUNCTION public.set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_documents_updated_at ON public.documents;
CREATE TRIGGER trg_documents_updated_at
    BEFORE UPDATE ON public.documents
    FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();
