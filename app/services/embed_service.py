"""
services/embed_service.py — D4 embedding step for the D3 ingestion pipeline.

Called by the separately deployable Pub/Sub ingestion worker after chunking.
Fetches all un-embedded chunks for a document, calls the embedding API in
batches of ``EMBED_BATCH``, and upserts the resulting vectors into
``public.chunk_vectors``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.db.alloydb import get_connection
from app.config import get_settings
from app.d4_vector_index.embeddings import _embed_texts_sync

logger = logging.getLogger(__name__)

EMBED_BATCH = 100  # chunks per API call (API limit for this model)


def embed_chunks_for_doc(doc_id: str, *, force: bool = False) -> int:
    """
    Synchronous entry point — safe to call from a thread-pool executor.

    Embeds every chunk in ``doc_id`` that does not yet have a row in
    ``chunk_vectors``, then upserts all vectors in a single batch INSERT.

    Parameters
    ----------
    doc_id : UUID string of the document to embed.

    Returns
    -------
    int
        Number of chunks embedded and written to ``chunk_vectors``.
    """
    model = get_settings().rag_embedding_model
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT c.chunk_id, c.text,
                   c.doc_id, c.jurisdiction, c.doc_type,
                   c.trust_level, c.effective_date, c.source_org
            FROM   chunks c
            LEFT JOIN chunk_vectors cv USING (chunk_id)
            WHERE  c.doc_id = %s
              AND  (%s OR cv.chunk_id IS NULL OR cv.embedding_model IS DISTINCT FROM %s)
            ORDER  BY c.chunk_index
            """,
            [doc_id, force, model],
        )
        rows = cur.fetchall()
        cur.close()

    if not rows:
        logger.info("[embed] doc_id=%s — all chunks already embedded", doc_id)
        return 0

    logger.info("[embed] doc_id=%s — embedding %d chunks via %s", doc_id, len(rows), model)

    total = 0
    for batch_start in range(0, len(rows), EMBED_BATCH):
        batch = rows[batch_start : batch_start + EMBED_BATCH]
        texts = [r[1] for r in batch]

        vectors = _embed_texts_sync(texts, task_type="RETRIEVAL_DOCUMENT")

        row_placeholders = []
        params: list = []
        for row, vec in zip(batch, vectors):
            chunk_id, _text, b_doc_id, jurisdiction, doc_type, trust_level, effective_date, source_org = row
            vec_str = "[" + ",".join(f"{v:.8f}" for v in vec) + "]"
            row_placeholders.append("(%s,%s,%s,%s,%s,%s,%s,%s::vector,%s,%s)")
            params += [
                str(chunk_id), str(b_doc_id),
                jurisdiction, doc_type,
                trust_level, effective_date, source_org,
                vec_str,
                model,
                datetime.now(timezone.utc),
            ]

        sql = (
            "INSERT INTO chunk_vectors "
            "(chunk_id, doc_id, jurisdiction, doc_type, trust_level, effective_date, source_org, embedding, embedding_model, embedded_at) "
            "VALUES " + ", ".join(row_placeholders) +
            " ON CONFLICT (chunk_id) DO UPDATE SET "
            "embedding = EXCLUDED.embedding, embedding_model = EXCLUDED.embedding_model, "
            "embedded_at = EXCLUDED.embedded_at, doc_id = EXCLUDED.doc_id, "
            "jurisdiction = EXCLUDED.jurisdiction, doc_type = EXCLUDED.doc_type, "
            "trust_level = EXCLUDED.trust_level, effective_date = EXCLUDED.effective_date, "
            "source_org = EXCLUDED.source_org"
        )

        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(sql, params)
            cur.close()

        total += len(batch)
        logger.info("[embed] doc_id=%s  progress=%d/%d", doc_id, total, len(rows))

    logger.info("[embed] doc_id=%s — done, %d vectors written", doc_id, total)
    return total


def vectors_complete_for_doc(doc_id: str) -> bool:
    """Every chunk must have a vector from the configured embedding space."""
    model = get_settings().rag_embedding_model
    with get_connection() as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT COUNT(*) AS chunks,
                       COUNT(*) FILTER (
                           WHERE cv.embedding IS NOT NULL AND cv.embedding_model = %s
                       ) AS compatible
                  FROM chunks c
                  LEFT JOIN chunk_vectors cv USING (chunk_id)
                 WHERE c.doc_id = %s
                """,
                (model, doc_id),
            )
            chunks, compatible = cur.fetchone()
        finally:
            cur.close()
    return chunks > 0 and chunks == compatible
