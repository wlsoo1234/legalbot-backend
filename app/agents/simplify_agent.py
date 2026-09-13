import json
from app.schemas import CleanedTextInput, SimplifiedResult
from app.llm import get_gemini_client, DEFAULT_MODEL


async def simplify_legalese(input_data: CleanedTextInput) -> SimplifiedResult:
    """
    4.0 Simplify Legalese Agent

    Takes cleaned text + sections from 3.0 Text Processing and produces
    a plain-language summary with extracted rights and obligations
    using the Gemini LLM.
    """

    client = get_gemini_client()

    prompt = f"""You are a legal document simplifier for Malaysian tenancy agreements.
Analyze the following tenancy agreement text and return a JSON object with exactly these fields:

1. "plain_summary": A clear, jargon-free summary of the entire agreement in 3-5 sentences.
   Write as if explaining to someone with no legal background.
2. "rights": A JSON array of strings listing each tenant RIGHT found in the agreement.
   Each right should be one clear sentence.
3. "obligations": A JSON array of strings listing each tenant OBLIGATION found in the agreement.
   Each obligation should be one clear sentence.

IMPORTANT: Return ONLY valid JSON, no markdown, no code fences, no extra text.

Agreement text:
\"\"\"
{input_data.text}
\"\"\"
"""

    if input_data.sections:
        sections_text = "\n".join(
            f"Section: {s.title}\n{s.content}" for s in input_data.sections
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

        return SimplifiedResult(
            plain_summary=data.get("plain_summary", "Unable to generate summary."),
            rights=data.get("rights", []),
            obligations=data.get("obligations", []),
        )

    except Exception as e:
        return SimplifiedResult(
            plain_summary=f"LLM analysis encountered an error: {str(e)}. Please try again.",
            rights=[],
            obligations=[],
        )
