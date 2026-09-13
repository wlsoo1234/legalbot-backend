import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.controllers import agreement_analysis_controller, agreement_generation_controller, legal_assistant_controller
from app.routers import agreement_processing, rag, d3
from app.rag_chatbot.router import router as chatbot_router
from app.logging_context import reset_request_id, set_request_id

logger = logging.getLogger(__name__)

app = FastAPI(
    title="LegalBot API",
    description=(
        "Multi-agent AI backend for detecting red flags in tenancy agreements. "
        "Modules 4.0 (Simplify Legalese) and 5.0 (ToS Gotcha Analyzer)."
    ),
    version="0.1.0",
)

app.include_router(d3.router)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = request_id
    token = set_request_id(request_id)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "request_failed request_id=%s method=%s path=%s duration_ms=%d",
            request_id,
            request.method,
            request.url.path,
            int((time.perf_counter() - started) * 1000),
        )
        raise
    finally:
        reset_request_id(token)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_complete request_id=%s method=%s path=%s status=%d duration_ms=%d",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        int((time.perf_counter() - started) * 1000),
    )
    return response


# CORS — allow all origins for hackathon speed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def health_check():
    return {
        "status": "ok",
        "service": "LegalBot API",
        "version": "0.1.0",
    }


# Mount routers
# app.include_router(simplify.router)
# app.include_router(gotcha.router)
# app.include_router(analyze.router)
# app.include_router(analyze_agreement.router)
# app.include_router(generate_draft.router)
app.include_router(agreement_processing.router)
app.include_router(rag.router)
app.include_router(chatbot_router)
# D4 remains an internal repository and is deliberately not mounted.
app.include_router(legal_assistant_controller.router)
app.include_router(agreement_analysis_controller.router)
app.include_router(agreement_generation_controller.router)
