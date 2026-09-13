"""
Step 3 of the ingestion pipeline: chunking — AGENTIC mode.

Pipeline
--------
1. Download extracted JSON from GCS.
2. Flatten pages → full text + page map.
3. Split at legal section headings (deterministic regex).
4. Sub-split any section > MAX_SECTION_CHARS via recursive character splitting.
5. Merge tiny fragments (< MIN_SECTION_CHARS) into their predecessor.
6. Send ALL section headings + text previews to Gemini Flash in ONE batched
   call → get refined section_title + one-sentence proposition per chunk.
7. Persist ChunkRecords in batches, updating chunk_count for live polling.
8. Mark document ready.

Fallback: if step 6 (Gemini call) fails for any reason, regex-detected
headings are used as-is and proposition is left blank — the pipeline
never fails because of an LLM error.
"""

from __future__ import annotations

import json
import hashlib
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from app.schemas_rag import ChunkRecord, LegalDocument, RegisteredSource
from app.services.gcs import download_bytes
from app.services.pipeline_store import mark_failed, update_document

logger = logging.getLogger(__name__)
CHUNK_NAMESPACE = uuid.UUID("62595967-6294-47c5-b73e-a5b2d366404f")


def deterministic_chunk_id(doc_id: str, chunk_index: int, text: str) -> str:
    """Stable ID makes retries safe across partially-completed chunk batches."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(CHUNK_NAMESPACE, f"{doc_id}:{chunk_index}:{digest}"))


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Parse 'gs://bucket/path/to/blob' → (bucket, blob_name)."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {uri!r}")
    without_scheme = uri[5:]
    bucket, _, blob = without_scheme.partition("/")
    return bucket, blob


# ── tunables ──────────────────────────────────────────────────────────────────

MAX_SECTION_CHARS = 3000   # sections larger than this are sub-split
MIN_SECTION_CHARS = 150    # sections smaller than this are merged into previous
OVERLAP_CHARS     = 100    # context overlap when sub-splitting large sections
MIN_EXTRACTED_CHARS = 200

ALLOWED_LANGUAGES: frozenset[str] = frozenset({
    "en", "ms", "zh", "zh-hans", "zh-hant", "ta",
})

# ── regex patterns ────────────────────────────────────────────────────────────

# Primary split: numbered provisions & major structural markers
_LEGAL_HEADING_RE = re.compile(
    r"^(?:"
    r"(?:PART|CHAPTER|SCHEDULE|ANNEX|APPENDIX)\s+[A-ZIVX\d]+"  # PART I, CHAPTER 2
    r"|(?:SECTION|ARTICLE)\s+\d+"                               # SECTION 1
    r"|\d{1,3}[A-Z]?[.]\s{1,8}[A-Z][a-z]"                     # 1.   Short title
    r")",
    re.MULTILINE,
)

_STRIP_FENCES_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


# ── validation ────────────────────────────────────────────────────────────────

class ValidationError(ValueError):
    """Raised when a pre-chunk validation rule is violated."""


def _validate_pre_chunk(source: RegisteredSource, pipeline_doc: dict) -> None:
    """Gate documents against D3 quality rules. Raises ValidationError on failure."""
    text_len = pipeline_doc.get("extracted_text_length") or 0
    if text_len < MIN_EXTRACTED_CHARS:
        raise ValidationError(
            f"extracted_text_length={text_len} is below the minimum "
            f"threshold of {MIN_EXTRACTED_CHARS} chars. "
            "The document may be empty, image-only, or extraction failed."
        )
    if not (source.jurisdiction or "").strip():
        raise ValidationError("Source is missing 'jurisdiction' — required for D3 indexing.")
    if not (source.doc_type or "").strip():
        raise ValidationError("Source is missing 'doc_type' — required for D3 indexing.")
    tl = getattr(source, "trust_level", None)
    if tl is None or not (1 <= tl <= 5):
        raise ValidationError(
            f"trust_level={tl!r} is out of the allowed range 1–5. "
            "1=government (highest), 5=unverified (lowest)."
        )
    lang = (getattr(source, "language", "en") or "en").lower()
    if lang not in ALLOWED_LANGUAGES:
        raise ValidationError(
            f"language={lang!r} is not in the allowed set {sorted(ALLOWED_LANGUAGES)}."
        )


# ── page helpers ──────────────────────────────────────────────────────────────

def _build_page_map(pages: list[dict]) -> list[tuple[int, int, int]]:
    """Return list of (char_start_global, char_end_global, page_number)."""
    mapping: list[tuple[int, int, int]] = []
    offset = 0
    for page in pages:
        text = page.get("text", "")
        mapping.append((offset, offset + len(text), page["page"]))
        offset += len(text) + 1  # +1 for the '\n' separator
    return mapping


def _char_to_page(char_offset: int, page_map: list[tuple[int, int, int]]) -> int:
    for start, end, page in page_map:
        if start <= char_offset < end:
            return page
    return page_map[-1][2] if page_map else 1


def _flatten_pages(pages: list[dict]) -> str:
    return "\n".join(p.get("text", "") for p in pages)


# ── structural splitting ──────────────────────────────────────────────────────

def _split_at_legal_headings(full_text: str) -> list[tuple[int, int, str]]:
    """
    Split full_text at every legal section heading detected by regex.
    Returns list of (char_start, char_end, heading_text).
    """
    matches = list(_LEGAL_HEADING_RE.finditer(full_text))

    if not matches:
        return [(0, len(full_text), "Document")]

    segments: list[tuple[int, int, str]] = []

    # Preamble text before first heading
    if matches[0].start() > MIN_SECTION_CHARS:
        segments.append((0, matches[0].start(), "Preamble"))

    for i, m in enumerate(matches):
        start = m.start()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        segments.append((start, end, m.group(0).strip()))

    return segments


def _recursive_split(
    text: str,
    base_offset: int,
    heading: str,
    page_map: list[tuple[int, int, int]],
) -> list[dict[str, Any]]:
    """
    Recursively split an oversized segment using paragraph → sentence → word
    boundaries until every piece is ≤ MAX_SECTION_CHARS.
    """
    if len(text) <= MAX_SECTION_CHARS:
        page_s = _char_to_page(base_offset, page_map)
        page_e = _char_to_page(base_offset + max(len(text) - 1, 0), page_map)
        return [{
            "text":       text.strip(),
            "char_start": base_offset,
            "char_end":   base_offset + len(text),
            "page_start": page_s,
            "page_end":   page_e,
            "raw_heading": heading,
        }]

    target   = MAX_SECTION_CHARS - OVERLAP_CHARS
    best_pos = -1
    for sep in ("\n\n", "\n", ". ", " "):
        pos = text.rfind(sep, target // 2, target)
        if pos != -1:
            best_pos = pos + len(sep)
            break
    if best_pos <= 0:
        best_pos = MAX_SECTION_CHARS

    result = _recursive_split(text[:best_pos], base_offset, heading, page_map)
    if text[best_pos:].strip():
        result += _recursive_split(
            text[best_pos:], base_offset + best_pos, heading + " (cont.)", page_map
        )
    return result


# ── Gemini agentic labeling ───────────────────────────────────────────────────

def _apply_fallback_labels(chunks: list[dict]) -> None:
    """Set section_title = raw_heading and proposition = '' for every chunk."""
    for c in chunks:
        c.setdefault("section_title", c.get("raw_heading") or "")
        c.setdefault("proposition", "")


def _label_with_gemini(raw_chunks: list[dict], doc_type: str) -> list[dict]:
    """
    One Gemini Flash call to label all chunks with:
      - section_title: clean heading (≤ 80 chars)
      - proposition:   one-sentence legal rule summary (≤ 30 words)

    Silently falls back if the LLM call fails for any reason — the
    pipeline is never blocked by an LLM error.
    """
    try:
        from app.llm import get_gemini_client  # noqa: PLC0415
        from app.config import get_settings  # noqa: PLC0415
        client = get_gemini_client()
    except Exception as exc:
        logger.warning("[agentic] Gemini client unavailable, skipping labeling: %s", exc)
        _apply_fallback_labels(raw_chunks)
        return raw_chunks

    # Build compact index: heading + first 300 chars of text per chunk
    index_lines: list[str] = []
    for i, c in enumerate(raw_chunks):
        preview = c["text"][:300].replace("\n", " ").strip()
        index_lines.append(
            f'[{i}] heading: "{c["raw_heading"]}" | preview: "{preview}"'
        )

    prompt = (
        f"You are a legal document analyst specialising in {doc_type} documents.\n\n"
        "Below is an indexed list of text segments from a legal document.\n"
        "For EACH segment return a JSON object with exactly two keys:\n"
        '  "section_title": a clean concise heading (≤ 80 chars) — '
        "refine the existing heading if good, or derive one if missing\n"
        '  "proposition": one sentence (≤ 30 words) stating the core legal '
        "rule, obligation, or subject this segment establishes\n\n"
        f"Return a JSON ARRAY with exactly {len(raw_chunks)} objects "
        "in the same order as the input. "
        "Output ONLY the JSON array — no markdown, no code fences, no extra text.\n\n"
        "SEGMENTS:\n" + "\n".join(index_lines)
    )

    try:
        response = client.models.generate_content(
            model=get_settings().rag_chat_model,
            contents=prompt,
        )
        raw    = _STRIP_FENCES_RE.sub("", response.text.strip()).strip()
        labels = json.loads(raw)

        if not isinstance(labels, list) or len(labels) != len(raw_chunks):
            logger.warning(
                "[agentic] Gemini returned %s labels for %d chunks — using fallback",
                len(labels) if isinstance(labels, list) else "?",
                len(raw_chunks),
            )
            _apply_fallback_labels(raw_chunks)
            return raw_chunks

        for chunk, label in zip(raw_chunks, labels):
            chunk["section_title"] = (
                label.get("section_title") or chunk["raw_heading"] or ""
            )[:200]
            chunk["proposition"] = (label.get("proposition") or "")[:300]

        logger.info("[agentic] labeled %d chunks via Gemini Flash", len(raw_chunks))

    except Exception as exc:
        logger.warning("[agentic] Gemini labeling failed (%s) — using regex headings", exc)
        _apply_fallback_labels(raw_chunks)

    return raw_chunks


# ── main chunk builder ────────────────────────────────────────────────────────

def _agentic_make_chunks(
    full_text: str,
    page_map: list[tuple[int, int, int]],
    doc_type: str,
) -> list[dict[str, Any]]:
    """
    Full agentic chunking pipeline:
      1. Split at legal headings         (deterministic)
      2. Sub-split oversized sections    (recursive, deterministic)
      3. Merge tiny fragments            (< MIN_SECTION_CHARS → absorbed by predecessor)
      3b. Sub-split any chunks that grew too large after merging
      3c. Extract clean heading from first line of each chunk
      4. Label with Gemini Flash         (one batched API call)
      5. Assign sequential chunk_index
    """
    # Steps 1 & 2
    segs       = _split_at_legal_headings(full_text)
    raw_chunks: list[dict] = []
    for start, end, heading in segs:
        seg_text = full_text[start:end]
        if not seg_text.strip():
            continue
        raw_chunks.extend(_recursive_split(seg_text, start, heading, page_map))

    if not raw_chunks:
        return []

    # Step 3: merge tiny fragments into predecessor
    merged: list[dict] = []
    for chunk in raw_chunks:
        if merged and len(chunk["text"]) < MIN_SECTION_CHARS:
            prev = merged[-1]
            prev["text"]     = (prev["text"] + "\n" + chunk["text"]).strip()
            prev["char_end"] = chunk["char_end"]
            prev["page_end"] = chunk["page_end"]
        else:
            merged.append(chunk)

    # Step 3b: sub-split any chunks that grew over the limit after merging
    post_merge: list[dict] = []
    for chunk in merged:
        if len(chunk["text"]) > MAX_SECTION_CHARS:
            post_merge.extend(
                _recursive_split(chunk["text"], chunk["char_start"], chunk["raw_heading"], page_map)
            )
        else:
            post_merge.append(chunk)

    # Step 3c: extract a clean heading from the first 1-2 non-empty lines
    # (legal PDFs often put the section number on line 1 and the title on line 2)
    for chunk in post_merge:
        lines = [ln.strip() for ln in chunk["text"].split("\n") if ln.strip()]
        if len(lines) >= 2 and len(lines[0]) <= 12:
            # e.g. "12." on line 1 + "What is a sound mind" on line 2
            heading = f"{lines[0]} {lines[1]}"
        elif lines:
            heading = lines[0]
        else:
            heading = chunk["raw_heading"]
        chunk["raw_heading"] = heading[:80]

    # Step 4: Gemini Flash labeling
    labeled = _label_with_gemini(post_merge, doc_type)

    # Step 5: assign sequential index
    for i, c in enumerate(labeled):
        c["chunk_index"] = i

    return labeled


# ── public entry point ────────────────────────────────────────────────────────

def chunk_document(
    doc_id: str,
    source: RegisteredSource,
    repo,  # LegalKnowledgeRepository (abstract)
) -> int:
    """
    Download extracted JSON, produce ChunkRecords via agentic chunking,
    persist via repo, mark document chunked.

    Returns the number of chunks written.
    Raises on any unrecoverable error (caller should mark_failed).
    """
    from app.services.pipeline_store import get_document as get_pipeline_doc  # noqa: PLC0415

    pipeline_doc = get_pipeline_doc(doc_id)
    if pipeline_doc is None:
        raise ValueError(f"Pipeline document not found: {doc_id}")

    gcs_uri_extracted = pipeline_doc.get("gcs_uri_extracted")
    if not gcs_uri_extracted:
        raise ValueError(
            f"Document {doc_id} has no extracted GCS URI; run extraction first."
        )

    _validate_pre_chunk(source, pipeline_doc)

    # Download extracted JSON from GCS
    bucket, blob = _parse_gcs_uri(gcs_uri_extracted)
    raw_bytes    = download_bytes(bucket, blob)
    extracted: dict = json.loads(raw_bytes.decode("utf-8"))

    pages: list[dict] = extracted.get("pages", [])
    if not pages:
        raise ValueError(f"Extracted document {doc_id} contains no pages.")

    full_text = _flatten_pages(pages)
    page_map  = _build_page_map(pages)

    logger.info(
        "[chunk] doc_id=%s  text_len=%d  starting agentic chunker",
        doc_id, len(full_text),
    )

    raw_chunks = _agentic_make_chunks(full_text, page_map, source.doc_type)

    if not raw_chunks:
        raise ValueError(f"No chunks produced for document {doc_id}.")

    logger.info("[chunk] doc_id=%s  produced %d chunks", doc_id, len(raw_chunks))

    # Persist LegalDocument row
    now = datetime.now(timezone.utc)
    legal_doc = LegalDocument(
        doc_id=doc_id,
        source_id=source.source_id,
        jurisdiction=getattr(source, "jurisdiction", None),
        doc_type=source.doc_type,
        provider=source.source_org,
        title=source.title,
        url=source.canonical_url or pipeline_doc.get("source_url"),
        created_at=now,
    )
    repo.save_document(legal_doc)

    # Build ChunkRecords — combine section_title + proposition for rich metadata
    chunk_records: list[ChunkRecord] = []
    for c in raw_chunks:
        token_est     = max(1, len(c["text"]) // 4)
        section_label = (c.get("section_title") or "").strip()
        proposition   = (c.get("proposition")   or "").strip()

        # e.g. "Section 7 — Proposer may revoke before acceptance is communicated."
        if proposition and proposition not in section_label:
            section_label = f"{section_label} — {proposition}".strip(" —")

        chunk_records.append(
            ChunkRecord(
                chunk_id=deterministic_chunk_id(doc_id, c["chunk_index"], c["text"]),
                doc_id=doc_id,
                chunk_index=c["chunk_index"],
                text=c["text"],
                section_title=section_label[:400] or None,
                jurisdiction=getattr(source, "jurisdiction", ""),
                doc_type=source.doc_type,
                language=getattr(source, "language", "en"),
                trust_level=getattr(source, "trust_level", 1),
                source_org=getattr(source, "source_org", None),
                effective_date=getattr(source, "effective_date", None),
                tags=list(getattr(source, "tags", [])),
                version_label=pipeline_doc.get("version_label"),
                token_estimate=token_est,
                url_fragment=None,
                char_start=c["char_start"],
                char_end=c["char_end"],
                page_start=c["page_start"],
                page_end=c["page_end"],
                created_at=now,
            )
        )

    # Persist in batches — single multi-row INSERT per batch (one round-trip)
    BATCH_SIZE = 200
    total = len(chunk_records)
    done  = 0
    update_document(doc_id, chunk_count=0)  # signal chunking has started

    for i in range(0, total, BATCH_SIZE):
        batch = chunk_records[i : i + BATCH_SIZE]
        repo.upsert_chunks(batch)
        done += len(batch)
        update_document(doc_id, chunk_count=done)
        logger.info("[chunk] doc_id=%s  progress=%d/%d", doc_id, done, total)

    update_document(doc_id, status="chunked")
    logger.info("[chunk] doc_id=%s  status=chunked  total_chunks=%d", doc_id, total)
    return total
