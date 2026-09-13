"""
db/alloydb.py — Synchronous AlloyDB connection pool via the Google AlloyDB
Python Connector (psycopg2 driver).

Why synchronous?
  The repository interface (legal_base.py) is currently synchronous to keep the
  in-memory dev implementation simple.  Switching to async later only requires
  changing this module + adding `async def` to the abstract interface —
  all router/service code stays the same.

Auth
----
  Local dev  : gcloud auth application-default login
  Cloud Run  : attach a service account with roles/alloydb.client +
               roles/alloydb.databaseUser

Required env vars (set in .env):
  ALLOYDB_INSTANCE_URI  projects/<P>/locations/<R>/clusters/<C>/instances/<I>
  ALLOYDB_DB            database name, e.g. legalbot
  ALLOYDB_USER          IAM principal or built-in user
  ALLOYDB_PASSWORD      (omit for IAM auth)
  ALLOYDB_IP_TYPE       PUBLIC | PRIVATE | PSC  (default: PUBLIC)
"""

from __future__ import annotations

import contextlib
import logging
from typing import Generator

import pg8000
import pg8000.dbapi
from google.cloud.alloydb.connector import Connector, IPTypes
from google.oauth2 import service_account

from app.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IP type helper
# ---------------------------------------------------------------------------

def _resolve_ip_type(value: str) -> IPTypes:
    """Convert env string like 'PUBLIC' → IPTypes.PUBLIC."""
    mapping = {
        "PUBLIC": IPTypes.PUBLIC,
        "PRIVATE": IPTypes.PRIVATE,
        "PSC": IPTypes.PSC,
    }
    return mapping.get(value.upper(), IPTypes.PUBLIC)


# ---------------------------------------------------------------------------
# Lazy singleton connector + pool
# ---------------------------------------------------------------------------

_connector: Connector | None = None
_pool: list = []  # simple free-connections list
_pool_lock = None  # threading.Lock, lazily created


def _get_connector() -> Connector:
    global _connector
    if _connector is None:
        sa_key = get_settings().alloydb_sa_key
        if sa_key:
            # Admin API credentials (AlloyDB instance discovery)
            admin_creds = service_account.Credentials.from_service_account_file(
                sa_key,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            # DB credentials (metadata exchange token — must use alloydb.login scope only)
            db_creds = service_account.Credentials.from_service_account_file(
                sa_key,
                scopes=["https://www.googleapis.com/auth/alloydb.login"],
            )
            _connector = Connector(credentials=admin_creds, db_credentials=db_creds)
        else:
            _connector = Connector()
    return _connector


def _make_conn():
    """
    Create a new pg8000 connection via the AlloyDB Python Connector.
    """
    s = get_settings()
    ip_type = _resolve_ip_type(getattr(s, "alloydb_ip_type", "PUBLIC"))
    conn = _get_connector().connect(
        s.alloydb_instance_uri,
        "pg8000",
        user=s.alloydb_user,
        password=s.alloydb_password or None,
        db=s.alloydb_db,
        ip_type=ip_type,
        enable_iam_auth=True,
    )
    conn.autocommit = False
    return conn


def get_pool() -> list:
    """
    Return the module-level free-connection pool, pre-filling it on first call.
    """
    global _pool, _pool_lock
    import threading
    if _pool_lock is None:
        _pool_lock = threading.Lock()
    with _pool_lock:
        if not _pool:
            s = get_settings()
            for _ in range(getattr(s, "alloydb_pool_min", 1)):
                _pool.append(_make_conn())
    return _pool


@contextlib.contextmanager
def get_connection() -> Generator:
    """
    Context manager: borrow a connection from the pool (or create a new one),
    yield it, then return it.  On exception the connection is rolled back.

    Usage::

        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
    """
    global _pool, _pool_lock
    import threading
    if _pool_lock is None:
        _pool_lock = threading.Lock()

    conn = None
    with _pool_lock:
        if _pool:
            conn = _pool.pop()
    if conn is None:
        conn = _make_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            conn = None  # broken connection, discard
        raise
    finally:
        if conn is not None:
            with _pool_lock:
                _pool.append(conn)


def healthcheck() -> bool:
    """
    Quick liveness check.  Returns True if a round-trip SELECT succeeds.
    Safe to call from a /healthz handler.
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
        return True
    except Exception as exc:
        logger.error("AlloyDB healthcheck failed: %s", exc)
        return False


def close_pool() -> None:
    """
    Gracefully shut down all pooled connections and the connector.
    Call this from a FastAPI `lifespan` shutdown handler in production.
    """
    global _pool, _connector
    for conn in _pool:
        try:
            conn.close()
        except Exception:
            pass
    _pool.clear()
    if _connector:
        _connector.close()
        _connector = None