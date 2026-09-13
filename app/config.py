from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "LegalBot API"
    app_env: str = "development"
    # Avoid the commonly exported DEBUG shell variable (e.g. DEBUG=release).
    debug: bool = Field(True, validation_alias="APP_DEBUG")
    gcp_project_id: str = ""
    gcp_location: str = "asia-southeast1"
    google_application_credentials: str = ""

    # Canonical RAG runtime configuration
    rag_enabled: bool = True
    rag_admin_api_key: str = ""
    rag_chat_model: str = "gemini-2.5-flash"
    rag_embedding_model: str = "models/gemini-embedding-001"
    rag_embedding_dimension: int = 768
    rag_history_turns: int = 6
    rag_vector_candidates: int = 20
    rag_lexical_candidates: int = 20
    rag_rrf_k: int = 60
    rag_max_context_chars: int = 24000
    rag_low_context_threshold: float = 0.45
    rag_max_upload_mb: int = 25
    rag_max_fetch_mb: int = 10
    rag_fetch_timeout_seconds: int = 30
    rag_ingest_topic: str = "legal-ingest"

    # -----------------------------------------------------------------------
    # GCS
    # -----------------------------------------------------------------------
    gcs_bucket: str = "legalbot-documents"
    # Path to a service-account key file for GCS (leave blank to use ADC)
    gcs_sa_key: str = ""

    # -----------------------------------------------------------------------
    # AlloyDB  (required when USE_ALLOYDB=true)
    # Format: projects/<P>/locations/<R>/clusters/<C>/instances/<I>
    # -----------------------------------------------------------------------
    use_alloydb: bool = True
    alloydb_instance_uri: str = ""
    alloydb_db: str = "legalbot"
    alloydb_user: str = ""
    alloydb_password: str = ""
    alloydb_ip_type: str = "PUBLIC"   # PUBLIC | PRIVATE | PSC
    alloydb_pool_min: int = 1
    alloydb_pool_max: int = 10
    # Path to a service-account key file for AlloyDB (kitahack-488509 project)
    alloydb_sa_key: str = ""
    # Optional direct asyncpg URL for D4 local dev (overrides AlloyDB connector)
    # Format: postgresql+asyncpg://user:password@host:5432/dbname
    async_database_url: str = ""

    # -----------------------------------------------------------------------
    # Google Cloud project (used by Pub/Sub, etc.)
    # -----------------------------------------------------------------------
    google_cloud_project: str = ""
    pubsub_topic_id: str = "agreement-analysis"

    @property
    def effective_gcp_project(self) -> str:
        return self.google_cloud_project or self.gcp_project_id


@lru_cache()
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_google_credentials():
    """Load configured service-account credentials lazily, or use ADC."""
    settings = get_settings()
    key_path = (
        settings.google_application_credentials
        or settings.gcs_sa_key
        or settings.alloydb_sa_key
    )
    if not key_path:
        return None
    path = Path(key_path)
    if not path.is_file():
        raise RuntimeError(f"Google credential file does not exist: {path}")
    from google.oauth2 import service_account

    return service_account.Credentials.from_service_account_file(
        str(path),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )


@lru_cache(maxsize=1)
def get_firestore_client():
    from google.cloud import firestore

    settings = get_settings()
    return firestore.Client(
        project=settings.effective_gcp_project or None,
        credentials=get_google_credentials(),
    )


@lru_cache(maxsize=1)
def get_gcs_client():
    from google.cloud import storage

    settings = get_settings()
    return storage.Client(
        project=settings.effective_gcp_project or None,
        credentials=get_google_credentials(),
    )


@lru_cache(maxsize=1)
def get_pubsub_client():
    from google.cloud import pubsub_v1

    return pubsub_v1.PublisherClient(credentials=get_google_credentials())


@lru_cache(maxsize=1)
def get_gemini_client():
    from google import genai

    settings = get_settings()
    return genai.Client(
        vertexai=True,
        project=settings.effective_gcp_project,
        location=settings.gcp_location,
        credentials=get_google_credentials(),
    )
