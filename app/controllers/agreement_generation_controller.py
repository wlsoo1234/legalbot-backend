from fastapi import APIRouter
from app.schemas import SendMessageRequest, CreateNewAgreementRequest

from app.services.agreement_generation_service import create_new_agreement, get_generated_agreement_details, get_generated_agreement_list, revert_agreement_audit, revise_agreement, submit_agreement_for_audit

router = APIRouter(prefix="/api/v1/agreements", tags=["Agreement generation"])

@router.post("/create")
async def create_new_agreement_route(request: CreateNewAgreementRequest):
    return await create_new_agreement(request=request)

@router.post("/revise/{agreement_id}")
async def revise_agreement_route(agreement_id: str, request: SendMessageRequest):
    return await revise_agreement(agreement_id=agreement_id, query=request.text)

@router.get("/created")
async def get_generated_agreement_list_route():
    return await get_generated_agreement_list()

@router.get("/created/{agreement_id}")
async def get_generated_agreement_details_route(agreement_id: str):
    return await get_generated_agreement_details(id=agreement_id);

@router.post("/created/{agreement_id}/audit")
async def submit_agreement_for_audit_route(agreement_id: str):
    return await submit_agreement_for_audit(agreement_id=agreement_id)

@router.post("/created/{agreement_id}/audit/revert")
async def revert_agreement_audit_route(agreement_id: str):
    return await revert_agreement_audit(agreement_id=agreement_id)