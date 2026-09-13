"""Compatibility export for the centrally configured lazy Gemini client."""

from app.config import get_gemini_client

DEFAULT_MODEL = "gemini-2.5-flash"

__all__ = ["DEFAULT_MODEL", "get_gemini_client"]
