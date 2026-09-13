from common.gemini import get_gemini_client
from app.config import get_settings
import asyncio


async def gemini_generate_content(contents, model=None, temperature=0.1, system_instruction=None):
    return await asyncio.to_thread(
        get_gemini_client().models.generate_content,
        model=model or get_settings().rag_chat_model,
        config={"system_instruction": system_instruction, "temperature": temperature},
        contents=contents,
    )
