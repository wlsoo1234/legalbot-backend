import json
import uuid
import time
import asyncio

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from app.schemas import AgreementProcessingResult, Status, SendMessageRequest, CreateNewAgreementRequest
from app.firestore import db
from google.cloud import firestore
from app.llm import get_gemini_client
from google.cloud.firestore_v1.query import Query
from app.cloud_clients import get_gcs_client, get_pubsub_client
from app.config import get_settings

router = APIRouter(prefix="/api/v1", tags=["Agreement Processing"])

async def _publish(topic_id: str, payload: dict) -> str:
    settings = get_settings()
    publisher = get_pubsub_client()
    project_id = settings.effective_gcp_project
    future = publisher.publish(
        publisher.topic_path(project_id, topic_id),
        data=json.dumps(payload).encode("utf-8"),
    )
    return await asyncio.to_thread(future.result, timeout=30)


# ──────────────────────────────────────────────
# POST /upload — Create record, fire background task, return FAST
# ──────────────────────────────────────────────
@router.post("/upload", response_model=AgreementProcessingResult)
async def upload_agreement(
    name: str = Form(..., description="User-given name for this agreement"),
    file: UploadFile = File(..., description="PDF file to upload"),
):
    """
    Upload a tenancy agreement PDF.
    1. Uploads the file to GCS.
    2. Saves the database record in Firestore.
    3. Simulates publishing to a message queue by starting a background task with the document ID.
    Returns IMMEDIATELY after queueing document processing.
    """

    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()
    doc_id = str(uuid.uuid4())
    timestamp = int(time.time() * 1000)

    # ── Step 1: Synchronous GCS Upload ──
    # loop = asyncio.get_event_loop()

    def upload_to_gcs():
        bucket_name = get_settings().gcs_bucket
        bucket = get_gcs_client().bucket(bucket_name)
        file_key = f"agreements/{doc_id}/{file.filename}"
        blob = bucket.blob(file_key)
        blob.upload_from_string(pdf_bytes, content_type="application/pdf")
        return f"gs://{bucket_name}/{file_key}"
        
    storage_uri = await asyncio.to_thread(upload_to_gcs)
    print(f"[{doc_id}] Uploaded to {storage_uri} synchronously.")

    # ── Step 2: Synchronous Firestore Record Creation ──
    doc_ref = db.collection("agreements").document(doc_id)
    doc_ref.set({
        "id": doc_id,
        "name": name,
        "storage_key": storage_uri,
        "createdAt": timestamp,
        "status": Status.PROCESSING.value,
        "extracted_text": None,
        "simplified": None,
        "gotcha": None,
        "key_terms": None,
        "decision": None,
        "error_message": None,
    })

    # ── Step 3: Trigger Background Processing (Simulating Message Queue) ──
    # We only pass the doc_id to simulate a message payload
    # asyncio.create_task(background_processing_task(doc_id))
    payload = {"doc_id": doc_id}
    await _publish(get_settings().pubsub_topic_id, payload)

    # ── Return immediately ──
    return AgreementProcessingResult(
        agreement_id=doc_id,
        status=Status.PROCESSING,
    )


@router.get("/agreements/uploaded")
async def get_uploaded_agreements():
    docs = db.collection("agreements").order_by("createdAt", direction=Query.DESCENDING).get()
    
    data = []
    for doc in docs:
        doct_dict = doc.to_dict()
        data.append({
            "id": doct_dict["id"],
            "name": doct_dict["name"],
            "status": doct_dict["status"],
            "createdAt": doct_dict["createdAt"],
        })
    return {
        "items": data
    }

@router.get("/agreements/uploaded/{id}")
async def get_uploaded_agreement_details(id: str):
    doc = db.collection("agreements").document(id).get()
    doc_dict = doc.to_dict();
    return {
        "id": doc_dict["id"],
        "name": doc_dict["name"],
        "status": doc_dict["status"],
        "createdAt": doc_dict["createdAt"],
        "summary": doc_dict["simplified"]["plain_summary"],
        "tenantRights": doc_dict["simplified"]["rights"],
        "obligations": doc_dict["simplified"]["obligations"],
        "redFlags": doc_dict["gotcha"]["red_flags"],
        "keyTerms": doc_dict["key_terms"],
        "decision": doc_dict["decision"],
    }

@router.get("/agreements/uploaded/{agreement_id}/chat")
async def get_agreement_assistant_chat(agreement_id: str):
    docs = db.collection("agreements").document(agreement_id).collection("chat").get()
    
    data = []
    for doc in docs:
        doct_dict = doc.to_dict()
        data.append({
            "id": doct_dict["id"],
            "text": doct_dict["text"],
            "role": doct_dict["role"],
            "createdAt": doct_dict["createdAt"],
        })
    return {
        "items": data
    }

async def get_chat_history(agreement_id: str):
    docs = db.collection("agreements").document(agreement_id).collection("chat").get()
    
    data = []
    for doc in docs:
        doct_dict = doc.to_dict()
        # The SDK expects 'parts' as a list of dictionaries
        data.append({
            "role": doct_dict["role"],
            "parts": [{"text": doct_dict["text"]}]
        })
    return data

@router.post("/agreements/uploaded/{agreement_id}/chat")
async def send_agreement_assistant_chat(agreement_id: str, request: SendMessageRequest):
    user_message_id = str(uuid.uuid4())
    db.collection("agreements").document(agreement_id).collection("chat").add({
        "id": user_message_id,
        "text": request.text,
        "role": "user",
        "createdAt": int(time.time() * 1000),
    })

    history = await get_chat_history(agreement_id)
    agreement_text = db.collection("agreements").document(agreement_id).get().to_dict()["extracted_text"]

    system_prompt = f"""
    You are a specialized Legal Assistant. Your goal is to provide accurate 
    answers based strictly on the provided agreement text.

    ### CONSTRAINTS:
    1. Only answer using the text provided below.
    2. If the answer is missing, say you don't know.
    3. Reference specific Clause or Section numbers.
    4. Response in markdown format.

    ### AGREEMENT TEXT:
    {agreement_text}
    """

    client = get_gemini_client()
    response = client.models.generate_content(
        model=get_settings().rag_chat_model,
        config={
            "system_instruction": system_prompt,
            "temperature": 0.1,
        },
        contents=[
            *history,
            {
                "role": "user", 
                "parts": [{"text": request.text}]
            }
        ]
    )

    responseTime = int(time.time() * 1000)
    assistant_message_id = str(uuid.uuid4())
    db.collection("agreements").document(agreement_id).collection("chat").add({
        "id": assistant_message_id,
        "text": response.text,
        "role": "model",
        "createdAt": responseTime,
    })

    return {
        "id": assistant_message_id,
        "text": response.text,
        "role": "model",
        "createdAt": responseTime
    }
    
@router.post("/agreements/create")
async def create_new_agreement(request: CreateNewAgreementRequest):
    topic_id = "agreement-write"
    agreement_id = str(uuid.uuid4())
    payload = {"task_type": "CREATE", "agreement_id": agreement_id, "address": request.address, "start_date": request.start_date, "duration": request.duration, "rent": request.rent, "special_clauses": request.special_clauses}
    await _publish(topic_id, payload)
    timestamp = int(time.time() * 1000)
    db.collection("created-agreement").document(agreement_id).set({
            "id": agreement_id,
            "name": request.name,
            "address": request.address,
            "status": "PENDING",
            "version": [],
            "updated_at": timestamp,
            "created_at": timestamp
        })

    return {
        "id": agreement_id,
        "status": "PENDING",
    }

@router.post("/agreements/revise/{agreement_id}")
async def revise_agreement(agreement_id: str, request: SendMessageRequest):
    topic_id = "agreement-write"
    payload = {"task_type": "REVISE", "agreement_id": agreement_id, "query": request.text}
    await _publish(topic_id, payload)

    timestamp = int(time.time() * 1000)
    doc_ref = db.collection("created-agreement").document(agreement_id)
    doc = doc_ref.get()
    doc_dict = doc.to_dict()
    versions = doc_dict.get("version", [])
    latest_version = sorted(versions, key=lambda x: x['created_at'])[-1]

    doc_ref.update({
        "version": firestore.ArrayUnion([{"agreement_content": latest_version["agreement_content"], "revision_query": request.text, "status": "PENDING", "created_at": timestamp}]),
    })

    return {
        "id": agreement_id
    }

@router.get("/agreements/generated")
async def get_generated_agreement_list():
    doc = db.collection("created-agreement").order_by("updated_at", direction=Query.DESCENDING).stream()
    data = []
    for doc in doc:
        doc_dict = doc.to_dict()
        data.append({
            "id": doc_dict["id"],
            "name": doc_dict["name"],
            "address": doc_dict["address"],
            "status": doc_dict["status"],
            "updatedAt": doc_dict["updated_at"],
        })
    return {
        "items": data
    }

@router.get("/agreements/generated/{agreement_id}")
async def get_generated_agreement_details(agreement_id: str):
    doc = db.collection("created-agreement").document(agreement_id).get()
    doc_dict = doc.to_dict()
    return {
        "id": doc_dict["id"],
        "name": doc_dict["name"],
        "address": doc_dict["address"],
        "status": doc_dict["status"],
        "version": doc_dict["version"],
        "updatedAt": doc_dict["updated_at"],
        "createdAt": doc_dict["created_at"],
    }

@router.post("/agreements/generated/{agreement_id}/audit")
async def submit_agreement_for_audit(agreement_id: str):
    db.collection("created-agreement").document(agreement_id).update({
        "status": "AUDIT",
    })

    return {
        "id": agreement_id,
        "status": "AUDIT",
    }

@router.post("/agreements/generated/{agreement_id}/audit/revert")
async def revert_agreement_audit(agreement_id: str):
    db.collection("created-agreement").document(agreement_id).update({
        "status": "DRAFT",
    })

    return {
        "id": agreement_id,
        "status": "DRAFT",
    }
