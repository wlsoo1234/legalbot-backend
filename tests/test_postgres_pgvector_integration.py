"""Optional PostgreSQL/pgvector integration test.

Set RAG_TEST_DATABASE_URL to enable. The test uses temporary tables only.
"""

import os

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_postgres_supports_vector_and_simple_full_text_search():
    database_url = os.getenv("RAG_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("RAG_TEST_DATABASE_URL is not configured")
    database_url = database_url.replace("postgresql+asyncpg://", "postgresql://", 1)

    import asyncpg

    connection = await asyncpg.connect(database_url)
    try:
        await connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await connection.execute(
            """
            CREATE TEMP TABLE rag_test_chunks (
                chunk_id text PRIMARY KEY,
                body text NOT NULL,
                search_vector tsvector GENERATED ALWAYS AS (
                    to_tsvector('simple'::regconfig, body)
                ) STORED,
                embedding vector(3)
            )
            """
        )
        await connection.executemany(
            "INSERT INTO rag_test_chunks (chunk_id, body, embedding) VALUES ($1, $2, $3::vector)",
            [
                ("semantic", "quiet enjoyment covenant", "[1,0,0]"),
                ("lexical", "rare exact statutory phrase", "[0,1,0]"),
            ],
        )
        vector_hit = await connection.fetchval(
            "SELECT chunk_id FROM rag_test_chunks ORDER BY embedding <=> '[1,0,0]'::vector LIMIT 1"
        )
        lexical_hit = await connection.fetchval(
            """
            SELECT chunk_id FROM rag_test_chunks
             WHERE search_vector @@ websearch_to_tsquery('simple', 'rare exact statutory phrase')
             ORDER BY ts_rank_cd(search_vector, websearch_to_tsquery('simple', 'rare exact statutory phrase')) DESC
             LIMIT 1
            """
        )
        assert vector_hit == "semantic"
        assert lexical_hit == "lexical"
    finally:
        await connection.close()
