"""
services/extract_service.py — Step 2: Download raw blob from GCS, extract
text, upload extracted JSON, update the documents row.

Extracted JSON format (written to GCS)
---------------------------------------
{
  "doc_id": "...",
  "pages": [
    {
      "page": 1,
      "text": "...",
      "section_hints": ["PART I — PRELIMINARY", ...]
    }
  ],
  "source": {
    "gcs_uri_raw": "gs://...",
    "source_id": "..."
  }
}

After this step the document status is ``extracted``.
Call chunk_service.chunk_document(doc_id) next.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section heading detection patterns (PDF text / plain HTML)
# Ordered most-specific → least-specific.
# ---------------------------------------------------------------------------
_HEADING_PATTERNS = [
    re.compile(r"^(?:PART|SECTION|ARTICLE|CHAPTER|SCHEDULE)\s+[IVXivx\d]+[\.\s—-]", re.MULTILINE),
    re.compile(r"^\d+\.\s{1,4}[A-Z][A-Z\s]{3,40}$", re.MULTILINE),
    re.compile(r"^[A-Z][A-Z\s]{4,50}$", re.MULTILINE),
]


def _detect_headings(text: str, max_hints: int = 8) -> list[str]:
    """Return up to *max_hints* likely section headings found in *text*."""
    seen: set[str] = set()
    hints: list[str] = []
    for pattern in _HEADING_PATTERNS:
        for m in pattern.finditer(text):
            heading = m.group(0).strip()[:120]
            if heading and heading not in seen:
                seen.add(heading)
                hints.append(heading)
                if len(hints) >= max_hints:
                    return hints
    return hints


# ---------------------------------------------------------------------------
# PDF extraction  (PyMuPDF / fitz)
# ---------------------------------------------------------------------------

def _extract_pdf(raw_bytes: bytes) -> list[dict]:
    """
    Extract text page-by-page from a PDF using PyMuPDF.

    Returns a list of page dicts::

        [{"page": 1, "text": "...", "section_hints": [...]}, ...]
    """
    import fitz  # PyMuPDF  # noqa: PLC0415

    pages: list[dict] = []
    doc = fitz.open(stream=raw_bytes, filetype="pdf")
    for i, page in enumerate(doc, start=1):
        # get_text("text") gives plain text; "blocks" gives layout-aware blocks
        block_texts: list[str] = []
        for block in page.get_text("blocks"):
            # block: (x0, y0, x1, y1, text, block_no, block_type)
            if block[6] == 0:  # type 0 = text block
                t = block[4].strip()
                if t:
                    block_texts.append(t)
        page_text = "\n\n".join(block_texts)
        pages.append({
            "page": i,
            "text": page_text,
            "section_hints": _detect_headings(page_text),
        })
    doc.close()
    return pages


# ---------------------------------------------------------------------------
# HTML extraction (BeautifulSoup)
# ---------------------------------------------------------------------------

def _extract_html(raw_bytes: bytes) -> list[dict]:
    """
    Extract main text content from an HTML page.

    Returns a single-element list (HTML docs don't have page numbers).
    """
    from bs4 import BeautifulSoup  # noqa: PLC0415

    html = raw_bytes.decode("utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")

    # Strip boilerplate
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
        tag.decompose()

    # Prefer <main> or <article>; fall back to <body>
    container = soup.find("main") or soup.find("article") or soup.find("body")
    if container is None:
        container = soup

    # Preserve paragraph spacing
    raw_text = container.get_text(separator="\n")
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", raw_text.strip())

    return [{
        "page": 1,
        "text": text,
        "section_hints": _detect_headings(text),
    }]


# ---------------------------------------------------------------------------
# GCS path helpers
# ---------------------------------------------------------------------------

def _split_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    """``gs://bucket/path/to/blob`` → ``(bucket, path/to/blob)``."""
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {gcs_uri!r}")
    remainder = gcs_uri[5:]
    bucket, _, blob = remainder.partition("/")
    return bucket, blob


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_document(doc_id: str) -> dict:
    """
    Step 2 pipeline entry point.

    1. Loads the documents row to get ``gcs_uri_raw`` and ``source_id``.
    2. Downloads the raw blob from GCS.
    3. Extracts text (PDF or HTML path).
    4. Uploads the extracted JSON to GCS.
    5. Updates the documents row → ``status='extracted'``.

    Returns the extracted-metadata dict::

        {
          "gcs_uri_extracted": "gs://...",
          "page_count": 12,
          "extracted_text_length": 48320
        }
    """
    from app.config import get_settings  # noqa: PLC0415
    from app.services.gcs import download_bytes, upload_text  # noqa: PLC0415
    from app.services.pipeline_store import get_document, update_document  # noqa: PLC0415

    s = get_settings()

    doc = get_document(doc_id)
    if doc is None:
        raise ValueError(f"documents row not found for doc_id={doc_id!r}")

    gcs_uri_raw: str = doc["gcs_uri_raw"]
    source_id: str = str(doc["source_id"])

    logger.info("[extract] starting doc_id=%s gcs_uri_raw=%s", doc_id, gcs_uri_raw)

    # Download raw
    bucket, blob_name = _split_gcs_uri(gcs_uri_raw)
    raw_bytes = download_bytes(bucket, blob_name)

    # Detect format from blob extension
    is_pdf = blob_name.lower().endswith(".pdf")

    pages = _extract_pdf(raw_bytes) if is_pdf else _extract_html(raw_bytes)

    total_text_length = sum(len(p["text"]) for p in pages)
    page_count = len(pages)

    extracted_payload = {
        "doc_id": doc_id,
        "pages": pages,
        "source": {
            "gcs_uri_raw": gcs_uri_raw,
            "source_id": source_id,
        },
    }

    # Upload extracted JSON
    extracted_blob = f"legal/extracted/{source_id}/{doc_id}.json"
    gcs_uri_extracted = upload_text(
        json.dumps(extracted_payload, ensure_ascii=False, indent=2),
        s.gcs_bucket,
        extracted_blob,
        prefix="",           # blob_name is already absolute
    )

    update_document(
        doc_id,
        status="extracted",
        gcs_uri_extracted=gcs_uri_extracted,
        page_count=page_count,
        extracted_text_length=total_text_length,
    )

    logger.info(
        "[extract] done doc_id=%s pages=%d chars=%d",
        doc_id, page_count, total_text_length,
    )

    return {
        "gcs_uri_extracted": gcs_uri_extracted,
        "page_count": page_count,
        "extracted_text_length": total_text_length,
    }
