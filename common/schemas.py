from pydantic import BaseModel, Field
from enum import Enum

class Status(str, Enum):
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class GenerateAgreementStatus(str, Enum):
    PENDING = "PENDING"
    DRAFT = "DRAFT"
    AUDIT = "AUDIT"
    VERIFIED = "VERIFIED"
    FAILED = "FAIELD"

class AgreementWriteTaskType(str, Enum):
    CREATE = "CREATA"
    REVISE = "REVISE"

class SendMessageRequest(BaseModel):
    """Request to send a message to the assistant."""
    text: str = Field(..., min_length=1, max_length=10000, description="Message to send to the assistant")
    session_id: str | None = Field(None, description="Conversation UUID; generated when absent")
    jurisdiction: str = Field("MY", min_length=2, max_length=32)
    doc_type: str | None = None
    language: str | None = None
    trust_level_max: int | None = Field(None, ge=1, le=5)
    top_k: int = Field(5, ge=1, le=20)

class CreateNewAgreementRequest(BaseModel):
    """Request to create a new agreement."""
    name: str = Field(..., description="Name of the agreement")
    address: str = Field(..., description="Property address")
    start_date: str = Field(..., description="Start date of the agreement")
    duration: int = Field(..., description="Duration of the agreement in months")
    rent: float = Field(..., description="Monthly rent amount")
    special_clauses: str | None = Field(None, description="Optional special clauses to include")
