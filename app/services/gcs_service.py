import time
import uuid
import asyncio
from fastapi import UploadFile
from common.gcs import get_gcs_client
from app.config import get_settings

async def upload_agreement_to_gcs(id: str, file: UploadFile):
        uid = str(uuid.uuid4())
        timestamp = int(time.time() * 1000)
        pdf_bytes = await file.read()

        bucket_name = get_settings().gcs_bucket
        bucket = get_gcs_client().bucket(bucket_name)
        file_key = f"agreements/{id}/{timestamp}-{uid}"
        blob = bucket.blob(file_key)
        await asyncio.to_thread(blob.upload_from_string, pdf_bytes, content_type="application/pdf")
        return f"gs://{bucket_name}/{file_key}"
