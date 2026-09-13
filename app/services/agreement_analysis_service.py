from fastapi import HTTPException, UploadFile

from app.repositories.agreement_repository import create_agreement, create_agreement_chat_message, select_agreement_by_id, select_agreement_chat_by_id, select_all_agreements_descending_by_created_at, update_agreement
from app.services.gcs_service import upload_agreement_to_gcs
from app.services.gemini_query_service import gemini_generate_content
from app.services.pubsub_service import publish_agreement_analysis_message
from common.prompts.agreement_analysis_prompts import get_agreement_analysis_prompt
from common.schemas import Status

async def upload_agreement(
    name: str,
    file: UploadFile
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    
    agreement = await create_agreement(
        name=name,
        status=Status.PROCESSING
    )

    agreement_id = agreement.get("id")
    storage_key = await upload_agreement_to_gcs(id=agreement_id, file=file)
    await update_agreement(id=agreement_id, storage_key=storage_key)

    await publish_agreement_analysis_message(agreement_id)

    return {
        "agreement_id": agreement_id,
        "status": Status.PROCESSING.value,
    }


async def get_uploaded_agreements():
    data = await select_all_agreements_descending_by_created_at()
    return {
        "items": data
    }

async def get_uploaded_agreement_details(id: str):
    return await select_agreement_by_id(id)

async def get_agreement_assistant_chat_history(id: str):
    data = await select_agreement_chat_by_id(id)
    return {
        "items": data
    }

async def send_agreement_assistant_message(id: str, text: str):
    await create_agreement_chat_message(id, text, "user")

    history = await select_agreement_chat_by_id(id)
    formatted_history = [
        {
            "role": item["role"],
            "parts": [{"text": item["text"]}]
        }
        for item in history
    ]
    agreement = await select_agreement_by_id(id)
    agreement_text = agreement["extractedText"]

    response = await gemini_generate_content(
        system_instruction=get_agreement_analysis_prompt(agreement_text=agreement_text),
        contents=[
            *formatted_history,
            {
                "role": "user", 
                "parts": [{"text": text}]
            }
        ]
    )

    return await create_agreement_chat_message(id, response.text, "model")