
from app.repositories.created_agreement_repository import create_created_agreement, select_all_created_agreements_descending_by_updated_at, select_created_agreement_by_id, update_created_agreement_status_by_id, update_created_agreement_version_by_id
from app.services.pubsub_service import publish_create_new_agreement_message, publish_revise_agreement_message
from common.schemas import CreateNewAgreementRequest, GenerateAgreementStatus


async def get_generated_agreement_list():
    data = await select_all_created_agreements_descending_by_updated_at()
    return {
        "items": data
    }

async def get_generated_agreement_details(id: str):
    return await select_created_agreement_by_id(id)

async def create_new_agreement(request: CreateNewAgreementRequest):
    await publish_create_new_agreement_message(data=request)
    agreement = await create_created_agreement(
        name=request.name,
        address=request.address,
        status=GenerateAgreementStatus.PENDING
    )

    return {
        "id": agreement['id'],
        "status": GenerateAgreementStatus.PENDING.value,
    }

async def revise_agreement(agreement_id: str, query: str):
    await publish_revise_agreement_message(id=agreement_id, query=query)
    agreement = await select_created_agreement_by_id(id=agreement_id)

    versions = agreement.get("version", [])
    latest_version = sorted(versions, key=lambda x: x['created_at'])[-1]
    await update_created_agreement_version_by_id(
        id=agreement_id,
        content=latest_version["agreement_content"],
        query=query,
        status=GenerateAgreementStatus.PENDING   
    )

    return {
        "id": agreement_id
    }


async def submit_agreement_for_audit(agreement_id: str):
    await update_created_agreement_status_by_id(id=agreement_id, status=GenerateAgreementStatus.AUDIT)
    
    return {
        "id": agreement_id,
        "status": GenerateAgreementStatus.AUDIT.value,
    }

async def revert_agreement_audit(agreement_id: str):
    await update_created_agreement_status_by_id(id=agreement_id, status=GenerateAgreementStatus.DRAFT)
    return {
        "id": agreement_id,
        "status": GenerateAgreementStatus.DRAFT.value,
    }
