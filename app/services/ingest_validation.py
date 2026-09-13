"""Validation and bounded fetching for untrusted ingestion inputs."""

from __future__ import annotations

import ipaddress
import socket
import time
from collections.abc import Callable
from urllib.parse import urljoin, urlsplit


class IngestValidationError(ValueError):
    """A terminal validation error that should not be retried."""


def validate_pdf(
    filename: str,
    content_type: str | None,
    data: bytes,
    *,
    max_bytes: int,
) -> None:
    if not filename.lower().endswith(".pdf"):
        raise IngestValidationError("Only .pdf uploads are supported.")
    allowed_types = {"application/pdf", "application/octet-stream"}
    if content_type and content_type.lower().split(";", 1)[0] not in allowed_types:
        raise IngestValidationError("The upload MIME type is not application/pdf.")
    if not data:
        raise IngestValidationError("The uploaded PDF is empty.")
    if not data.startswith(b"%PDF-"):
        raise IngestValidationError("The upload does not have a valid PDF signature.")
    if len(data) > max_bytes:
        raise IngestValidationError(
            f"The uploaded PDF exceeds the {max_bytes // (1024 * 1024)} MB limit."
        )


def _is_public_address(value: str) -> bool:
    address = ipaddress.ip_address(value)
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def validate_public_url(
    url: str,
    *,
    resolver: Callable[..., list] = socket.getaddrinfo,
) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise IngestValidationError("URL scheme must be http or https.")
    if not parsed.hostname or parsed.username or parsed.password:
        raise IngestValidationError("URL must contain a public host and no credentials.")
    try:
        addresses = {
            item[4][0]
            for item in resolver(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
        }
    except (OSError, socket.gaierror) as exc:
        raise IngestValidationError(f"URL host could not be resolved: {exc}") from exc
    if not addresses or any(not _is_public_address(value) for value in addresses):
        raise IngestValidationError("URL resolves to a private or unsafe network address.")
    return url


def fetch_public_url(
    url: str,
    *,
    timeout_seconds: int,
    max_bytes: int,
    max_redirects: int = 5,
) -> tuple[bytes, str, str]:
    """Fetch with DNS/redirect SSRF checks and a streaming size limit."""
    import httpx

    current = url
    deadline = time.monotonic() + timeout_seconds
    with httpx.Client(timeout=timeout_seconds, follow_redirects=False) as client:
        for redirect_count in range(max_redirects + 1):
            validate_public_url(current)
            with client.stream("GET", current, headers={"User-Agent": "LegalBot-Ingest/1.0"}) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    if redirect_count >= max_redirects:
                        raise IngestValidationError("URL exceeded the redirect limit.")
                    location = response.headers.get("location")
                    if not location:
                        raise IngestValidationError("URL redirect did not include a destination.")
                    current = urljoin(current, location)
                    continue
                try:
                    response.raise_for_status()
                except httpx.HTTPError as exc:
                    raise IngestValidationError(f"URL fetch failed: {exc}") from exc
                content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                if content_type not in {"text/html", "text/plain", "application/xhtml+xml"}:
                    raise IngestValidationError(
                        f"URL returned unsupported content type {content_type or 'unknown'}."
                    )
                body = bytearray()
                for part in response.iter_bytes():
                    if time.monotonic() > deadline:
                        raise IngestValidationError("URL fetch exceeded the configured timeout.")
                    body.extend(part)
                    if len(body) > max_bytes:
                        raise IngestValidationError("URL content exceeds the configured download limit.")
                return bytes(body), content_type or "text/html", current
    raise IngestValidationError("URL fetch failed after redirects.")
