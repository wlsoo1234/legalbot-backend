"""
embeddings.py — Embedding provider for the D4 vector index.

Public API
----------
``embed_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]``
    Embed a single string.  Returns a 768-d float vector.

``embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]``
    Embed a batch of strings in one API call.  Returns one vector per input.

Model: configured Vertex AI embedding model (768 dimensions)

Task types
----------
``RETRIEVAL_DOCUMENT``  — use when embedding chunks at ingest time.
``RETRIEVAL_QUERY``     — use when embedding a user question for retrieval.
``SEMANTIC_SIMILARITY`` — use for general-purpose similarity comparisons.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

EMBEDDING_DIM: int = 768
# Maximum texts per embed_content call (API limit for this model)
_BATCH_LIMIT: int = 100


def _embed_texts_sync(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    """
    Synchronous implementation — calls the google-genai SDK directly.
    Handles batching internally so callers don't need to worry about API limits.
    """
    from app.llm import get_gemini_client  # lazy import to avoid circular deps
    from app.config import get_settings

    client = get_gemini_client()
    settings = get_settings()
    model = settings.rag_embedding_model
    dimension = settings.rag_embedding_dimension
    if dimension != EMBEDDING_DIM:
        raise RuntimeError(
            f"RAG_EMBEDDING_DIMENSION must be {EMBEDDING_DIM}; got {dimension}."
        )
    results: list[list[float]] = []

    for batch_start in range(0, len(texts), _BATCH_LIMIT):
        batch = texts[batch_start : batch_start + _BATCH_LIMIT]
        response = client.models.embed_content(
            model=model,
            contents=batch,
            config={
                "task_type": task_type,
                "output_dimensionality": dimension,
            },
        )
        for emb in response.embeddings:
            values = list(emb.values)
            if len(values) != dimension:
                raise RuntimeError(
                    f"Embedding model {model} returned {len(values)} dimensions; expected {dimension}."
                )
            results.append(values)

    logger.debug(
        "embed_texts: embedded %d text(s) via %s task_type=%s",
        len(texts),
        model,
        task_type,
    )
    return results


async def embed_text(
    text: str,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[float]:
    """
    Embed a single string using the configured Vertex AI embedding model.

    Runs the synchronous SDK call in a thread-pool executor so it never
    blocks the uvicorn event loop.

    Parameters
    ----------
    text      : The text to embed.
    task_type : ``"RETRIEVAL_DOCUMENT"`` (default) or ``"RETRIEVAL_QUERY"``.

    Returns
    -------
    list[float]
        A 768-dimensional float vector.
    """
    loop = asyncio.get_running_loop()
    vectors = await loop.run_in_executor(None, _embed_texts_sync, [text], task_type)
    return vectors[0]


async def embed_texts(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    """
    Embed a batch of strings using the configured Vertex AI embedding model.

    Batches are split into chunks of at most ``_BATCH_LIMIT`` texts per API
    call.  Runs synchronously in a thread-pool executor.

    Parameters
    ----------
    texts     : List of strings to embed.
    task_type : ``"RETRIEVAL_DOCUMENT"`` (default) or ``"RETRIEVAL_QUERY"``.

    Returns
    -------
    list[list[float]]
        One 768-dimensional vector per input string, in the same order.
    """
    if not texts:
        return []
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _embed_texts_sync, texts, task_type)
