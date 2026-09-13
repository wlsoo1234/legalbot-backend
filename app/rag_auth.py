"""Authentication dependency for RAG administration endpoints."""

from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import Header, HTTPException, status

from app.config import get_settings


async def require_rag_admin(
    api_key: Annotated[str | None, Header(alias="X-Admin-API-Key")] = None,
) -> None:
    configured_key = get_settings().rag_admin_api_key
    if not configured_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG administration is unavailable: RAG_ADMIN_API_KEY is not configured.",
        )
    if api_key is None or not hmac.compare_digest(api_key, configured_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid RAG administration API key.",
        )
