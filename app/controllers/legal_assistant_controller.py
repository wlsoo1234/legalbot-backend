from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_async_session
from app.services.legal_assistant_service import (
    get_consultation_history,
    send_consultation_message,
)
from common.schemas import SendMessageRequest

router = APIRouter(prefix="/api/v1/assistant/consultation", tags=["Legal Assistant"])


@router.get("/history")
async def get_consultation_history_route(
    session_id: str = Query(..., min_length=1, max_length=128),
):
    return await get_consultation_history(session_id)


@router.post("/send")
async def send_consultation_message_route(
    request: SendMessageRequest,
    db: AsyncSession = Depends(get_async_session),
):
    return await send_consultation_message(request, db)
