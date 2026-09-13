from fastapi import APIRouter, File, UploadFile, Form
from app.schemas import AgreementProcessingResult

from app.services.agreement_analysis_service import get_agreement_assistant_chat_history, get_uploaded_agreement_details, get_uploaded_agreements, send_agreement_assistant_message, upload_agreement
from common.schemas import SendMessageRequest

router = APIRouter(prefix="/api/v1/agreements/analysis", tags=["Agreement Analysis"])

@router.post("/upload", response_model=AgreementProcessingResult)
async def upload_agreement_route(
    name: str = Form(..., description="User-given name for this agreement"),
    file: UploadFile = File(..., description="PDF file to upload"),
):
    return await upload_agreement(name=name, file=file)

@router.get("/list")
async def get_uploaded_agreements_route():
    return await get_uploaded_agreements()

@router.get("/{agreement_id}")
async def get_uploaded_agreement_details_route(agreement_id: str):
    return await get_uploaded_agreement_details(id=agreement_id)

@router.get("/{agreement_id}/chat/message/history")
async def get_agreement_assistant_chat_history_route(agreement_id: str):
    return await get_agreement_assistant_chat_history(id=agreement_id)

@router.post("/{agreement_id}/chat/message/send")
async def send_agreement_assistant_message_route(agreement_id: str, request: SendMessageRequest):
    return await send_agreement_assistant_message(id=agreement_id, text=request.text)