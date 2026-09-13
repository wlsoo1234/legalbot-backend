"""
session.py — Shared async SQLAlchemy session factory.

Re-exports the AlloyDB-backed async session generator from the D4 vector
index module so that any other module (e.g. rag_chatbot) can depend on a
single, shared connection pool without duplicating the engine creation logic.

Usage (FastAPI dependency injection)::

    from app.db.session import get_async_session
    from sqlalchemy.ext.asyncio import AsyncSession

    @router.post("/example")
    async def endpoint(session: AsyncSession = Depends(get_async_session)):
        ...
"""

from app.d4_vector_index.repository import get_db as get_async_session  # noqa: F401

__all__ = ["get_async_session"]
