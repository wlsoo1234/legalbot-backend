import json
from app.schemas import CleanedTextInput, GotchaResult, RedFlag, Severity
from app.llm import get_gemini_client, DEFAULT_MODEL


async def analyze_gotchas(input_data: CleanedTextInput) -> GotchaResult:
    """
    5.0 ToS Gotcha Analyzer Agent (Rules + LLM)

    Uses Gemini LLM to perform deep analysis of tenancy agreement clauses,
    identifying red flags, unfair terms, and hidden gotchas.
    """

    client = get_gemini_client()

    prompt = f"""You are a legal clause analyzer specializing in Malaysian tenancy agreements
and tenant protection. Analyze the following agreement text for red flags, unfair clauses,
and hidden risks that could harm the tenant.

For each problematic clause found, provide:
- "clause": The exact or paraphrased problematic text
- "severity": One of "low", "medium", "high", or "critical"
- "risk_score": A float from 0.0 to 1.0 (low=0.25, medium=0.5, high=0.75, critical=0.95)
- "explanation": Why this clause is problematic for the tenant
- "suggestion": Specific actionable advice for the tenant

Return a JSON object with exactly these fields:
{{
  "red_flags": [list of red flag objects as described above],
  "overall_risk_score": a float from 0.0 to 1.0 representing the overall agreement risk,
  "highlighted_clauses": [list of exact clause text strings that need attention]
}}

Focus on these categories:
- Unfair deposit/forfeiture terms
- One-sided termination clauses
- Excessive penalties or fees
- Landlord liability exclusions
- Privacy/access violations
- Hidden costs or charges
- Automatic renewal traps
- Waiver of statutory tenant rights
- Unreasonable maintenance obligations
- Discriminatory clauses

IMPORTANT: Return ONLY valid JSON, no markdown, no code fences, no extra text.
If no red flags are found, return empty arrays and a low risk score.

Agreement text:
\"\"\"
{input_data.text}
\"\"\"
"""

    if input_data.sections:
        sections_text = "\n".join(
            f"[Section: {s.title}]\n{s.content}" for s in input_data.sections
        )
        prompt += f"\n\nParsed sections:\n{sections_text}"

    try:
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
        )
        raw = response.text.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        data = json.loads(raw)

        red_flags = []
        for rf in data.get("red_flags", []):
            severity_str = rf.get("severity", "medium").lower()
            if severity_str not in ("low", "medium", "high", "critical"):
                severity_str = "medium"

            red_flags.append(
                RedFlag(
                    clause=rf.get("clause", "Unknown clause"),
                    severity=Severity(severity_str),
                    risk_score=max(0.0, min(1.0, float(rf.get("risk_score", 0.5)))),
                    explanation=rf.get("explanation", ""),
                    suggestion=rf.get("suggestion", ""),
                )
            )

        overall_risk = max(0.0, min(1.0, float(data.get("overall_risk_score", 0.5))))
        highlighted_clauses = data.get("highlighted_clauses", [])

        return GotchaResult(
            red_flags=red_flags,
            overall_risk_score=round(overall_risk, 2),
            highlighted_clauses=highlighted_clauses,
        )

    except Exception as e:
        return GotchaResult(
            red_flags=[
                RedFlag(
                    clause="Error during analysis",
                    severity=Severity.LOW,
                    risk_score=0.0,
                    explanation=f"LLM analysis encountered an error: {str(e)}",
                    suggestion="Please try again or check the API key configuration.",
                )
            ],
            overall_risk_score=0.0,
            highlighted_clauses=[],
        )
