"""Application configuration loaded from environment variables / .env file."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration for the application.

    Values are read from environment variables first, then from a .env file.
    The app will raise a ValidationError at startup if required fields are missing.
    """

    # ── Required ──────────────────────────────────────────────────────────────
    openai_api_key: str

    # ── Optional with defaults ─────────────────────────────────────────────────
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    chroma_persist_dir: str = "chroma_db"
    log_level: str = "INFO"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  # OPENAI_API_KEY and openai_api_key both work
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    The @lru_cache means the .env file is read exactly once per process,
    not on every request. FastAPI's Depends() will call this repeatedly,
    so caching is important for performance.
    """
    return Settings()
