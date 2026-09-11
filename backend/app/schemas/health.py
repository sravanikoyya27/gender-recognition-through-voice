"""Health-check response schema."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    # `model_loaded` intentionally uses the "model_" prefix that pydantic reserves.
    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "examples": [
                {
                    "status": "ok",
                    "app": "Gender Voice Recognition API",
                    "version": "1.0.0",
                    "environment": "development",
                    "database": "connected",
                    "model_loaded": True,
                    "timestamp": "2026-09-11T20:00:00Z",
                }
            ]
        },
    )

    status: Literal["ok", "degraded"]
    app: str
    version: str
    environment: str
    database: Literal["connected", "unavailable"]
    model_loaded: bool
    timestamp: datetime
