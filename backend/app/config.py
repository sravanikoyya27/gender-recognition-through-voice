"""Application configuration loaded from environment variables / .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent

_INSECURE_JWT_DEFAULT = "change-me-in-production-use-a-long-random-string"


class Settings(BaseSettings):
    """Typed settings. Values come from the environment, then `.env`, then defaults."""

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    APP_NAME: str = "Gender Voice Recognition API"
    APP_DESCRIPTION: str = "API for gender classification from voice"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    LOG_DIR: Path = BACKEND_DIR / "logs"
    API_V1_PREFIX: str = "/api/v1"

    # Comma-separated list of allowed browser origins (the frontend domain).
    CORS_ORIGINS: str = "http://localhost:3000"

    # --- Database ---
    DATABASE_URL: str = "postgresql+psycopg2://voice:voice@localhost:5432/voice_gender"

    # --- AWS ---
    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str | None = None

    # --- Auth (JWT) ---
    JWT_SECRET_KEY: str = _INSECURE_JWT_DEFAULT  # must be at least 32 characters
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_DAYS: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 90

    # --- Caching ---
    REDIS_URL: str | None = None  # e.g. redis://localhost:6379/0; unset -> in-memory cache
    CACHE_TTL_SECONDS: int = 3600

    # --- ML artifacts ---
    # Deployment copies written by ml_models/model_comparison.py next to model_config.json (the manifest).
    MODEL_DIR: Path = BACKEND_DIR / "app" / "ml" / "models"
    MODEL_MANIFEST_PATH: Path = BACKEND_DIR / "app" / "ml" / "models" / "model_config.json"
    MODEL_PATH: Path = BACKEND_DIR / "app" / "ml" / "models" / "model_mlp.h5"
    SCALER_PATH: Path = BACKEND_DIR / "app" / "ml" / "models" / "scaler_mlp.pkl"
    # Phase 2D comparison report; used to seed the model_metrics table (optional).
    MODEL_COMPARISON_REPORT_PATH: Path = (
        BACKEND_DIR.parent / "ml_models" / "models" / "comparison" / "comparison_report.json"
    )

    # --- Uploads / predictions ---
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_AUDIO_EXTENSIONS: str = ".wav,.mp3,.webm,.ogg,.flac"
    MIN_AUDIO_SECONDS: float = 1.0
    MAX_AUDIO_SECONDS: float = 10.0
    PREDICTIONS_REQUIRE_AUTH: bool = True  # set false to allow anonymous predictions (demo)
    MAX_BATCH_FILES: int = 10

    # --- Derived helpers ---
    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def allowed_audio_extensions(self) -> set[str]:
        return {e.strip().lower() for e in self.ALLOWED_AUDIO_EXTENSIONS.split(",") if e.strip()}

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @field_validator("LOG_LEVEL", mode="before")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        return value.upper() if isinstance(value, str) else value

    @field_validator("JWT_SECRET_KEY")
    @classmethod
    def _secret_long_enough(cls, value: str) -> str:
        if len(value) < 32:
            raise ValueError("JWT_SECRET_KEY must be at least 32 characters long")
        return value

    @model_validator(mode="after")
    def _guard_production_secrets(self) -> "Settings":
        if self.is_production and self.JWT_SECRET_KEY == _INSECURE_JWT_DEFAULT:
            raise ValueError("JWT_SECRET_KEY must be set to a strong value when ENVIRONMENT=production")
        return self


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance (call `get_settings.cache_clear()` in tests to reload)."""
    return Settings()
