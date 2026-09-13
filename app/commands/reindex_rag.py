"""Re-embed chunks that are missing or use a different embedding model."""

from __future__ import annotations

import argparse
import logging

from app.config import get_settings
from app.db.alloydb import get_connection
from app.services.embed_service import embed_chunks_for_doc, vectors_complete_for_doc

logger = logging.getLogger(__name__)


def documents_to_reindex(*, force: bool = False) -> list[str]:
    model = get_settings().rag_embedding_model
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                SELECT DISTINCT c.doc_id::text
                  FROM chunks c
                  LEFT JOIN chunk_vectors cv USING (chunk_id)
                 WHERE %s OR cv.embedding IS NULL OR cv.embedding_model IS DISTINCT FROM %s
                 ORDER BY c.doc_id::text
                """,
                (force, model),
            )
            return [row[0] for row in cursor.fetchall()]
        finally:
            cursor.close()


def run(*, force: bool = False) -> int:
    doc_ids = documents_to_reindex(force=force)
    total = 0
    for index, doc_id in enumerate(doc_ids, start=1):
        logger.info("reindex_document doc_id=%s progress=%d/%d", doc_id, index, len(doc_ids))
        total += embed_chunks_for_doc(doc_id, force=force)
        if not vectors_complete_for_doc(doc_id):
            raise RuntimeError(f"Vector compatibility check failed for doc_id={doc_id}")
    print(f"Reindexed {total} chunks across {len(doc_ids)} documents using {get_settings().rag_embedding_model}.")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-embed every indexed chunk.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(force=args.force)


if __name__ == "__main__":
    main()
