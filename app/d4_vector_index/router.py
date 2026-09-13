"""
router.py — FastAPI router for the D4 vector index.

Endpoints
---------
POST /d4/embed
    Store or update the embedding for an existing chunk.

POST /d4/retrieve
    Top-K cosine-similarity search with optional metadata filters.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.d4_vector_index.embeddings import embed_text
from app.d4_vector_index.repository import get_db
from app.d4_vector_index.schemas import (
    EmbedRequest,
    EmbedResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from app.d4_vector_index.service import D4VectorService

router = APIRouter(prefix="/d4", tags=["D4 Vector Index"])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# POST /d4/embed
# ---------------------------------------------------------------------------


@router.post(
    "/embed",
    response_model=EmbedResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Store or update an embedding for a chunk",
    description=(
        "Upserts the 768-d embedding vector for an existing chunk. "
        "The chunk must have been ingested via the D3 pipeline first."
    ),
)
async def embed_chunk(
    body: EmbedRequest,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> EmbedResponse:
    """
    Request body::

        {
          "chunk_id": "uuid",
          "embedding": [0.12, -0.03, ...],   // 768 floats
          "model": "text-embedding-004"
        }

    Response (201)::

        { "chunk_id": "uuid", "status": "stored" }
    """
    svc = D4VectorService(session)
    try:
        await svc.store_embedding(
            chunk_id=body.chunk_id,
            embedding=body.embedding,
            model=body.model,
        )
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )
    except Exception as exc:
        logger.exception("store_embedding failed for chunk_id=%s", body.chunk_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store embedding: {exc}",
        )
    return EmbedResponse(chunk_id=body.chunk_id)


# ---------------------------------------------------------------------------
# POST /d4/retrieve
# ---------------------------------------------------------------------------


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    summary="Top-K cosine-similarity retrieval",
    description=(
        "Returns the top-K most similar chunks to the query embedding. "
        "Supports optional metadata filters (jurisdiction, doc_type, "
        "language, trust_max)."
    ),
)
async def retrieve_chunks(
    body: RetrieveRequest,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> RetrieveResponse:
    """
    Request body::

        {
          "query_embedding": [...],   // 768 floats
          "top_k": 8,
          "filters": {
              "jurisdiction": "MY",
              "doc_type": "deposit",
              "language": "en",
              "trust_max": 2
          }
        }

    Response (200)::

        {
          "results": [
            { "chunk_id": "...", "score": 0.83 },
            ...
          ]
        }

    ``score`` is the cosine similarity in [0, 1]; higher means more similar.
    """
    # Resolve query vector — embed text if no pre-computed vector provided
    if body.query_embedding is not None:
        query_vec = body.query_embedding
    else:
        try:
            query_vec = await embed_text(body.query_text, task_type="RETRIEVAL_QUERY")
        except Exception as exc:
            logger.exception("embed_text failed for query")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to embed query: {exc}",
            )

    svc = D4VectorService(session)
    try:
        results = await svc.retrieve_top_k(
            query_embedding=query_vec,
            top_k=body.top_k,
            jurisdiction=body.filters.jurisdiction,
            doc_type=body.filters.doc_type,
            language=body.filters.language,
            trust_max=body.filters.trust_max,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )
    except Exception as exc:
        logger.exception("retrieve_top_k failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {exc}",
        )
    return RetrieveResponse(results=results)
