"""Application configuration loaded from environment variables / .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent

_INSECURE_JWT_DEFAULT = "change-me-in-production-use-a-long-random-string"

#: Substrings that identify example / placeholder secrets (compared case-insensitively).
_PLACEHOLDER_MARKERS = (
    "change-me",
    "changeme",
    "replace-with",
    "replace_with",
    "your_secret",
    "your-secret",
    "your_password",
    "your_access_key",
    "your_secret_key",
    "example",
    "placeholder",
    "dummy",
    "insecure",
)
#: Example values shipped in the .env.example files and docs; never valid in production.
_KNOWN_PLACEHOLDER_SECRETS = frozenset(
    {
        _INSECURE_JWT_DEFAULT,
        "replace-with-a-long-random-string-of-32-plus-chars",
        "your_secret_key_at_least_32_chars",
        "gender-voice-bucket",
        "your_access_key",
        "your_secret_key",
        "your_password",
    }
)
_MIN_DISTINCT_SECRET_CHARS = 8


def looks_like_placeholder(value: str | None) -> bool:
    """True for empty values, known example values or values carrying an example marker."""
    if not value:
        return False
    lowered = value.strip().lower()
    return lowered in _KNOWN_PLACEHOLDER_SECRETS or any(marker in lowered for marker in _PLACEHOLDER_MARKERS)


def is_weak_secret(value: str) -> bool:
    """Short, placeholder-like or low-variety secrets (e.g. one repeated character)."""
    return len(value) < 32 or looks_like_placeholder(value) or len(set(value)) < _MIN_DISTINCT_SECRET_CHARS


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
    # S3 storage of uploads is OFF unless S3_ENABLED is true AND a bucket is configured.
    # A bucket name alone (e.g. the example value) never enables uploads.
    S3_ENABLED: bool = False
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
    # Phase 2C feature importances and Phase 2B training statistics (optional, for the results UI).
    FEATURE_IMPORTANCE_PATH: Path = (
        BACKEND_DIR.parent / "ml_models" / "models" / "sklearn" / "feature_importance.json"
    )
    TRAINING_METRICS_PATH: Path = BACKEND_DIR.parent / "ml_models" / "models" / "metrics_mlp.json"

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

    @property
    def jwt_secret_is_weak(self) -> bool:
        """True when the JWT secret is the default, an example value or low-variety (never logs it)."""
        return is_weak_secret(self.JWT_SECRET_KEY)

    @property
    def s3_storage_enabled(self) -> bool:
        return bool(self.S3_ENABLED and self.S3_BUCKET_NAME)

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

    @field_validator("S3_BUCKET_NAME", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", mode="before")
    @classmethod
    def _blank_placeholders(cls, value: str | None) -> str | None:
        """Treat empty strings as unset (the .env.example files leave these blank)."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _guard_secrets(self) -> "Settings":
        # Error messages deliberately never include the offending value.
        if self.is_production and self.jwt_secret_is_weak:
            raise ValueError(
                "JWT_SECRET_KEY is the default, an example value or too predictable; set a random secret "
                "of at least 32 characters when ENVIRONMENT=production"
            )
        if self.S3_ENABLED:
            if not self.S3_BUCKET_NAME:
                raise ValueError("S3_ENABLED=true requires S3_BUCKET_NAME")
            for name in ("S3_BUCKET_NAME", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
                if looks_like_placeholder(getattr(self, name)):
                    raise ValueError(f"{name} is an example value; S3 cannot be enabled with it")
        return self


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance (call `get_settings.cache_clear()` in tests to reload)."""
    return Settings()
