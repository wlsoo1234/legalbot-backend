from pydantic import BaseModel, Field
from enum import Enum


# --- Input from 3.0 Text Processing ---

class Section(BaseModel):
    """A section extracted from the tenancy agreement by 3.0 Text Processing."""
    title: str = Field(..., description="Section heading, e.g. 'Deposit', 'Termination'")
    content: str = Field(..., description="Full text content of the section")
    clause_spans: list[str] = Field(
        default_factory=list,
        description="Individual clause text spans within this section"
    )


class CleanedTextInput(BaseModel):
    """Input received from 3.0 Text Processing — cleaned text + sections."""
    text: str = Field(..., description="Full cleaned agreement text")
    sections: list[Section] = Field(
        default_factory=list,
        description="Parsed sections from the agreement"
    )


# --- Output of 4.0 Simplify Legalese ---

class SimplifiedResult(BaseModel):
    """Plain-language summary output from 4.0 Simplify Legalese agent."""
    plain_summary: str = Field(..., description="Agreement summarized in plain language")
    rights: list[str] = Field(default_factory=list, description="Tenant rights identified")
    obligations: list[str] = Field(default_factory=list, description="Tenant obligations identified")


# --- Output of 5.0 ToS Gotcha Analyzer ---

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Status(str, Enum):
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class RedFlag(BaseModel):
    """A single red flag detected by the 5.0 Gotcha Analyzer."""
    clause: str = Field(..., description="The problematic clause text")
    severity: Severity = Field(..., description="Severity level")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Risk score from 0 to 1")
    explanation: str = Field(..., description="Why this clause is problematic")
    suggestion: str = Field(..., description="Recommended action for the tenant")


class GotchaResult(BaseModel):
    """Full output from the 5.0 ToS Gotcha Analyzer agent."""
    red_flags: list[RedFlag] = Field(default_factory=list, description="Detected red flags")
    overall_risk_score: float = Field(0.0, ge=0.0, le=1.0, description="Overall agreement risk score")
    highlighted_clauses: list[str] = Field(
        default_factory=list,
        description="Clauses that need tenant attention"
    )

class AgreementProcessingResult(BaseModel):
    """Full output from the agreement processing."""
    agreement_id: str = Field(..., description="Unique ID for uploaded agreement")
    status: Status = Field(..., description="Status of the agreement processing")

class SendMessageRequest(BaseModel):
    """Request to send a message to the assistant."""
    text: str = Field(..., description="Message to send to the assistant")

class CreateNewAgreementRequest(BaseModel):
    """Request to create a new agreement."""
    name: str = Field(..., description="Name of the agreement")
    address: str = Field(..., description="Property address")
    start_date: str = Field(..., description="Start date of the agreement")
    duration: int = Field(..., description="Duration of the agreement in months")
    rent: float = Field(..., description="Monthly rent amount")
    special_clauses: str | None = Field(None, description="Optional special clauses to include")

# --- Combined response ---

class FullAnalysisResponse(BaseModel):
    """Combined response from both 4.0 and 5.0 agents."""
    simplified: SimplifiedResult
    gotcha: GotchaResult


# --- Analyze Agreement (frontend endpoint) ---

class AnalyzeAgreementRequest(BaseModel):
    """Request to analyze a tenancy agreement text."""
    text: str = Field(..., description="Full agreement text to analyze")


class KeyTerms(BaseModel):
    """Key financial terms extracted from the agreement."""
    monthly_rent: float = Field(..., description="Monthly rent amount")
    security_deposit_months: int = Field(..., description="Security deposit in months")


class RiskAssessment(BaseModel):
    """A single risk assessment item."""
    title: str = Field(..., description="Risk item title")
    risk_level: str = Field(..., description="Risk level: 'High Risk', 'Medium Risk', or 'Low Risk'")
    description: str = Field(..., description="Description of the risk")


class SigningDecision(BaseModel):
    """Overall signing decision for the agreement."""
    title: str = Field(..., description="Decision title")
    status: str = Field(..., description="Status: 'Safe', 'Review Required', or 'Do Not Sign'")
    description: str = Field(..., description="Decision explanation")


class AnalyzeAgreementResponse(BaseModel):
    """Full analysis response for the frontend."""
    report_id: str = Field(..., description="Unique ID for this analysis — use with /export-report/{report_id}")
    key_terms: KeyTerms
    risk_assessments: list[RiskAssessment]
    decision: SigningDecision



# --- Generate Draft (frontend endpoint) ---

class DraftAgreementRequest(BaseModel):
    """Request to generate a draft tenancy agreement."""
    agreement_name: str = Field(..., description="Name/title of the agreement")
    property_address: str = Field(..., description="Property address")
    duration_months: int = Field(..., description="Agreement duration in months")
    monthly_rent: float = Field(..., description="Monthly rent amount")
    special_clauses: str | None = Field(None, description="Optional special clauses to include")


class DraftAgreementResponse(BaseModel):
    """Generated draft agreement."""
    draft_agreement: str = Field(..., description="Full draft agreement text")
