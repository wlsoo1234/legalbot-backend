import uuid
import time
from common.firestore import get_firestore_client
from google.cloud.firestore_v1.query import Query

from common.schemas import Status

AGREEMENT_COLLECTION = "agreements"
CHAT_SUBCOLLECTION = "chat"

async def select_all_agreements_descending_by_created_at():
    docs = get_firestore_client().collection(AGREEMENT_COLLECTION).order_by("createdAt", direction=Query.DESCENDING).get()
    data = []
    for doc in docs:
        doct_dict = doc.to_dict()
        data.append({
            "id": doct_dict["id"],
            "name": doct_dict["name"],
            "status": doct_dict["status"],
            "createdAt": doct_dict["createdAt"],
        })
    return data

async def select_agreement_by_id(id: str):
    doc = get_firestore_client().collection(AGREEMENT_COLLECTION).document(id).get()
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
        "extractedText": doc_dict["extracted_text"],
    }

async def select_agreement_chat_by_id(id: str):
    docs = get_firestore_client().collection(AGREEMENT_COLLECTION).document(id).collection(CHAT_SUBCOLLECTION).get()
    
    data = []
    for doc in docs:
        doct_dict = doc.to_dict()
        data.append({
            "id": doct_dict["id"],
            "text": doct_dict["text"],
            "role": doct_dict["role"],
            "createdAt": doct_dict["createdAt"],
        })
    return data

async def create_agreement_chat_message(id: str, text: str, role: str):
    message_id = str(uuid.uuid4())
    created_at = int(time.time() * 1000)

    new_message = {
        "id": message_id,
        "text": text,
        "role": role,
        "createdAt": created_at
    }

    get_firestore_client().collection(AGREEMENT_COLLECTION).document(id).collection(CHAT_SUBCOLLECTION).add(new_message)

    return new_message

async def create_agreement(name: str, 
                           status: Status, 
                           storage_key: str = None, 
                           extracted_text: str = None, 
                           simplified = None, 
                           gotcha = None, 
                           key_terms = None, 
                           decision = None, 
                           error_message = None):
    agreement_id = str(uuid.uuid4())
    created_at = int(time.time() * 1000)

    new_agreement = {
        "id": agreement_id,
        "name": name,
        "storage_key": storage_key,
        "status": status.value,
        "extracted_text": extracted_text,
        "simplified": simplified,
        "gotcha": gotcha,
        "key_terms": key_terms,
        "decision": decision,
        "error_message": error_message,
        "createdAt": created_at,
    }

    get_firestore_client().collection(AGREEMENT_COLLECTION).document(agreement_id).set(new_agreement)
    return new_agreement

async def update_agreement(
                           id: str,
                           name: str = None, 
                           status: Status = None, 
                           storage_key: str = None, 
                           extracted_text: str = None, 
                           simplified = None, 
                           gotcha = None, 
                           key_terms = None, 
                           decision = None, 
                           error_message = None):
    
    update_data = {
        "name": name,
        "status": status,
        "storage_key": storage_key,
        "extracted_text": extracted_text,
        "simplified": simplified,
        "gotcha": gotcha,
        "key_terms": key_terms,
        "decision": decision,
        "error_message": error_message,
    }

    filtered_updates = {k: v for k, v in update_data.items() if v is not None}

    if filtered_updates:
        get_firestore_client().collection(AGREEMENT_COLLECTION).document(id).update(filtered_updates)
        return True
    return False
