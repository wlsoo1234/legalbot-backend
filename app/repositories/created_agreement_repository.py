import time
import uuid

from common.firestore import get_firestore_client
from google.cloud import firestore
from google.cloud.firestore_v1.query import Query

from common.schemas import GenerateAgreementStatus

CREATED_AGREEMENT_COLLECTION = "created-agreement"
CHAT_SUBCOLLECTION = "chat"

async def select_all_created_agreements_descending_by_updated_at():
    docs = get_firestore_client().collection(CREATED_AGREEMENT_COLLECTION).order_by("updated_at", direction=Query.DESCENDING).stream()
    data = []
    for doc in docs:
        doc_dict = doc.to_dict()
        data.append({
            "id": doc_dict["id"],
            "name": doc_dict["name"],
            "address": doc_dict["address"],
            "status": doc_dict["status"],
            "updatedAt": doc_dict["updated_at"],
        })
    return data;

async def select_created_agreement_by_id(id: str):
    doc = get_firestore_client().collection(CREATED_AGREEMENT_COLLECTION).document(id).get()
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

async def update_created_agreement_status_by_id(id: str, status: GenerateAgreementStatus):
    get_firestore_client().collection(CREATED_AGREEMENT_COLLECTION).document(id).update({
        "status": status.value,
    })

    return True

async def update_created_agreement_version_by_id(id: str, content: str, query: str, status: GenerateAgreementStatus):
    created_at = int(time.time() * 1000)
    get_firestore_client().collection(CREATED_AGREEMENT_COLLECTION).document(id).update({
            "version": firestore.ArrayUnion([
                {"agreement_content": content, 
                 "revision_query": query, 
                 "status": status.value, 
                 "created_at": created_at}
            ]),
        })

async def create_created_agreement(
            name: str,
            address: str,
            status: GenerateAgreementStatus,
            version = []
        ):
    uid = str(uuid.uuid4())
    timestamp = int(time.time() * 1000)
    new_agreement = {
            "id": uid,
            "name": name,
            "address": address,
            "status": status.value,
            "version": version,
            "updated_at": timestamp,
            "created_at": timestamp
        }
    get_firestore_client().collection(CREATED_AGREEMENT_COLLECTION).document(uid).set(new_agreement)
    return new_agreement
