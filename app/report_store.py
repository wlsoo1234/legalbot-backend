"""
In-memory report store for the hackathon.
Stores analysis results keyed by UUID so they can be retrieved for PDF export.

NOTE: This is in-memory only — data is lost on server restart.
For production, replace with a real database (e.g., D1 Session DB from the architecture).
"""

from collections import OrderedDict
from typing import Optional
from datetime import datetime

# Max reports to keep in memory (prevents unbounded growth during demo)
MAX_REPORTS = 100

_store: OrderedDict[str, dict] = OrderedDict()


def save_report(report_id: str, data: dict) -> None:
    """Save an analysis result to the in-memory store."""
    if len(_store) >= MAX_REPORTS:
        # Evict oldest entry (FIFO)
        _store.popitem(last=False)
    _store[report_id] = {
        **data,
        "created_at": datetime.utcnow().isoformat(),
    }


def get_report(report_id: str) -> Optional[dict]:
    """Retrieve an analysis result by report ID. Returns None if not found."""
    return _store.get(report_id)


def report_exists(report_id: str) -> bool:
    return report_id in _store
