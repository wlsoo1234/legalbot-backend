"""Async structured-output wrapper around the shared Vertex AI client."""

from __future__ import annotations

import asyncio
import json
import logging
from functools import partial
from typing import TypeVar

from pydantic import BaseModel

from app.config import get_settings
from app.llm import get_gemini_client
from app.rag_chatbot.schemas import AnswerJSON, QueryRewriteJSON

logger = logging.getLogger(__name__)
SchemaT = TypeVar("SchemaT", bound=BaseModel)


def _generate_structured_sync(
    system_prompt: str,
    user_prompt: str,
    schema: type[SchemaT],
    temperature: float,
) -> SchemaT:
    from google.genai import types

    response = get_gemini_client().models.generate_content(
        model=get_settings().rag_chat_model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=temperature,
        ),
    )
    raw = response.text or "{}"
    return schema.model_validate(json.loads(raw))


async def ask(system_prompt: str, user_prompt: str) -> AnswerJSON:
    return await asyncio.to_thread(
        _generate_structured_sync,
        system_prompt,
        user_prompt,
        AnswerJSON,
        0.1,
    )


async def rewrite_query(system_prompt: str, user_prompt: str) -> str:
    result = await asyncio.to_thread(
        _generate_structured_sync,
        system_prompt,
        user_prompt,
        QueryRewriteJSON,
        0.0,
    )
    return result.search_query.strip()
