import uuid
import time
import asyncio
from common.firestore import get_firestore_client

CONSULTATION_COLLECTION = "consultation"

async def select_all_consultation_history(session_id: str):
    def _read():
        from google.cloud.firestore_v1.base_query import FieldFilter

        docs = get_firestore_client().collection(CONSULTATION_COLLECTION).where(
            filter=FieldFilter("sessionId", "==", session_id)
        ).get()
        data = []
        for doc in docs:
            item = doc.to_dict()
            data.append({
                "id": item["id"],
                "text": item["text"],
                "role": item["role"],
                "createdAt": item["createdAt"],
                "sessionId": item["sessionId"],
            })
        return sorted(data, key=lambda item: item["createdAt"])
    return await asyncio.to_thread(_read)

async def create_consultation_message(text: str, role: str, session_id: str):
    uid = str(uuid.uuid4())
    createdAt = int(time.time() * 1000)
    new_message = {
        "id": uid,
        "text": text,
        "role": role,
        "createdAt": createdAt,
        "sessionId": session_id,
    }

    await asyncio.to_thread(
        get_firestore_client().collection(CONSULTATION_COLLECTION).add,
        new_message,
    )

    return new_message
