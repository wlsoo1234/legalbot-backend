import json
import asyncio

from common.pubsub import get_pubsub_client
from common.schemas import AgreementWriteTaskType, CreateNewAgreementRequest
from app.config import get_settings
AGREEMENT_ANALYSIS_TOPIC_ID = "agreement-analysis"
AGREEMENT_WRITE_TOPIC_ID = "agreement-write"

async def publish_agreement_analysis_message(id: str):
    pubsub_client = get_pubsub_client()
    settings = get_settings()
    topic_path = pubsub_client.topic_path(settings.effective_gcp_project, settings.pubsub_topic_id)
    payload = {"doc_id": id}
    payload_json = json.dumps(payload)
    future = pubsub_client.publish(topic_path, data=payload_json.encode("utf-8"))
    return await asyncio.to_thread(future.result, timeout=30)

async def publish_revise_agreement_message(id: str, query: str):
    pubsub_client = get_pubsub_client()
    settings = get_settings()
    topic_path = pubsub_client.topic_path(settings.effective_gcp_project, AGREEMENT_WRITE_TOPIC_ID)
    payload = {"task_type": AgreementWriteTaskType.REVISE.value, "agreement_id": id, "query": query}
    payload_json = json.dumps(payload)
    future = pubsub_client.publish(topic_path, data=payload_json.encode("utf-8"))
    return await asyncio.to_thread(future.result, timeout=30)

async def publish_create_new_agreement_message(id: str, data: CreateNewAgreementRequest):
    pubsub_client = get_pubsub_client()
    settings = get_settings()
    topic_path = pubsub_client.topic_path(settings.effective_gcp_project, AGREEMENT_WRITE_TOPIC_ID)
    payload = {
        "task_type": AgreementWriteTaskType.CREATE.value, 
        "agreement_id": id, 
        "address": data.address, 
        "start_date": data.start_date, 
        "duration": data.duration, 
        "rent": data.rent, 
        "special_clauses": data.special_clauses
    }
    payload_json = json.dumps(payload)
    future = pubsub_client.publish(topic_path, data=payload_json.encode("utf-8"))
    return await asyncio.to_thread(future.result, timeout=30)
