"""FastAPI application: metadata, middleware, lifecycle, error handling, health and routers.

Middleware order (outermost first): CORS -> request logging -> FastAPI/Starlette
exception handling -> routes. CORS is outermost so that error responses and
preflight requests carry the CORS headers as well.

Lifecycle uses the ``lifespan`` context (the FastAPI 0.104 replacement for the
deprecated ``@app.on_event("startup"/"shutdown")`` hooks):

startup   configure logging, log the configuration (secrets masked), probe the
          database and apply migrations (creates missing tables), verify the
          model artifacts exist and load them once, warm up feature extraction
shutdown  release the model, dispose database connections, remove temp files

Run directly with ``python -m app.main`` or ``uvicorn app.main:app``.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import api_router
from app.config import Settings, get_settings
from app.ml.inference import InferenceService
from app.models.database import SessionLocal, check_database_connection, engine, init_db
from app.schemas.health import HealthResponse
from app.services.analytics import seed_model_metrics
from app.services.prediction import warm_up_feature_extraction
from app.utils.audio import cleanup_temp_files
from app.utils.logger import configure_logging

logger = logging.getLogger("app")

#: Paths whose requests are logged in reduced form (no query string; bodies are never logged anywhere).
SENSITIVE_PATH_SUFFIXES = ("/auth/login", "/auth/register", "/auth/refresh")

OPENAPI_TAGS = [
    {"name": "system", "description": "Health and service information."},
    {"name": "authentication", "description": "Registration, login, token refresh and the current user."},
    {"name": "predictions", "description": "Gender prediction from uploaded audio (single and batch)."},
    {"name": "analytics", "description": "Per-user history and statistics; public model performance."},
]


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def _log_configuration(settings: Settings) -> None:
    """Log the effective configuration with secrets masked."""
    masked = settings.model_dump()
    for key in ("JWT_SECRET_KEY", "AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID"):
        if masked.get(key):
            masked[key] = "***"
    url = str(masked.get("DATABASE_URL", ""))
    if "@" in url and "://" in url:
        scheme, rest = url.split("://", 1)
        creds, host = rest.rsplit("@", 1)
        user = creds.split(":", 1)[0]
        masked["DATABASE_URL"] = f"{scheme}://{user}:***@{host}"
    if masked.get("REDIS_URL") and "@" in str(masked["REDIS_URL"]):
        masked["REDIS_URL"] = "redis://***"
    logger.info("Configuration: %s", {k: str(v) for k, v in masked.items()})


def _verify_artifacts(settings: Settings) -> None:
    for label, path in (
        ("model", settings.MODEL_PATH),
        ("scaler", settings.SCALER_PATH),
        ("manifest", settings.MODEL_MANIFEST_PATH),
    ):
        if path.exists():
            logger.info("Artifact %s: %s (%d bytes)", label, path, path.stat().st_size)
        else:
            logger.warning("Artifact %s missing: %s", label, path)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    configure_logging(settings.LOG_LEVEL, settings.LOG_DIR)
    logger.info("Starting %s v%s (%s)", settings.APP_NAME, settings.APP_VERSION, settings.ENVIRONMENT)
    _log_configuration(settings)

    # --- database ---------------------------------------------------------
    app.state.database_available = check_database_connection(cache_seconds=0)
    if app.state.database_available:
        logger.info("Database connection OK")
        try:
            logger.info("Database schema: %s", init_db())  # applies migrations / creates missing tables
            with SessionLocal() as db:
                seeded = seed_model_metrics(db)
                logger.info("Model metrics table: %s", f"{seeded} rows seeded" if seeded else "up to date")
        except Exception:  # noqa: BLE001 - a migration failure must be visible but not fatal here
            logger.exception("Database schema initialisation failed; persistence may be unavailable")
            app.state.database_available = False
    else:
        logger.warning("Database not reachable at startup; continuing without persistence")

    # --- model ------------------------------------------------------------
    _verify_artifacts(settings)
    inference = InferenceService(settings.MODEL_PATH, settings.SCALER_PATH, settings.MODEL_MANIFEST_PATH)
    try:
        inference.load()
    except Exception:  # noqa: BLE001 - a bad artifact must not take the API down
        logger.exception("Failed to load model artifacts")
    app.state.inference = inference
    try:
        warm_up_feature_extraction()
    except Exception:  # noqa: BLE001 - warm-up is an optimisation only
        logger.exception("Feature extraction warm-up failed")

    app.state.started_at = datetime.now(UTC)
    logger.info(
        "Startup complete: database=%s model_loaded=%s docs=/docs",
        app.state.database_available,
        inference.is_loaded,
    )

    yield

    # --- shutdown ---------------------------------------------------------
    logger.info("Shutting down")
    inference.unload()
    engine.dispose()
    removed = cleanup_temp_files()
    logger.info("Shutdown complete (%d temporary files removed)", removed)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        description=settings.APP_DESCRIPTION,
        version=settings.APP_VERSION,
        docs_url="/docs",  # Swagger UI
        redoc_url="/redoc",  # ReDoc
        openapi_tags=OPENAPI_TAGS,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.database_available = False
    app.state.inference = InferenceService(
        settings.MODEL_PATH, settings.SCALER_PATH, settings.MODEL_MANIFEST_PATH
    )

    # --- middleware (registered inner to outer; CORS last so it is outermost) ----
    @app.middleware("http")
    async def logging_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Log method, path, status and response time for every request.

        Sensitive endpoints are logged without their query string; request and
        response bodies are never logged.
        """
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        path = request.url.path
        sensitive = path.endswith(SENSITIVE_PATH_SUFFIXES)
        shown = path if sensitive or not request.url.query else f"{path}?{request.url.query}"
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            # The traceback is logged once by the global exception handler.
            logger.error("%s %s -> 500 (%.1f ms) id=%s", request.method, shown, elapsed_ms, request_id)
            raise
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        logger.info(
            "%s %s -> %s (%.1f ms) id=%s%s",
            request.method,
            shown,
            response.status_code,
            elapsed_ms,
            request_id,
            " [sensitive]" if sensitive else "",
        )
        return response

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
        expose_headers=[],
        max_age=3600,
    )

    # --- error handlers: every error body is {"detail": "<message>"} --------------
    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> Response:
        """422 (400 on auth routes) with a readable message; field details in ``errors``."""
        messages = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", []) if p != "body")
            msg = err.get("msg", "invalid value").removeprefix("Value error, ")
            messages.append(f"{loc}: {msg}" if loc else msg)
        detail = "; ".join(messages) or "Validation error"
        if request.url.path.startswith(f"{settings.API_V1_PREFIX}/auth/"):
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": detail})
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": detail, "errors": exc.errors()},
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> Response:
        """4xx/5xx raised by routes or routing (404) with a string ``detail``."""
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        headers = getattr(exc, "headers", None) or None
        return JSONResponse(status_code=exc.status_code, content={"detail": detail}, headers=headers)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> Response:
        """Anything unexpected: log the full traceback, hide internals from the client.

        Starlette runs this handler outside the CORS middleware, so the CORS
        headers for an allowed origin are added here explicitly.
        """
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        headers: dict[str, str] = {}
        origin = request.headers.get("origin")
        if origin and (origin in settings.cors_origins or "*" in settings.cors_origins):
            headers = {
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Credentials": "true",
                "Vary": "Origin",
            }
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
            headers=headers,
        )

    # --- system endpoints ---------------------------------------------------------
    @app.get("/health", response_model=HealthResponse, tags=["system"], summary="Health check")
    async def health() -> HealthResponse:
        """Liveness/readiness probe for load balancers. No authentication.

        ``status`` is ``ok`` when the database is reachable and the model is
        loaded, otherwise ``degraded`` (the process is up but not fully functional).
        """
        db_ok = await run_in_threadpool(check_database_connection)
        app.state.database_available = db_ok
        model_ok = app.state.inference.is_loaded
        return HealthResponse(
            status="ok" if (db_ok and model_ok) else "degraded",
            app=settings.APP_NAME,
            version=settings.APP_VERSION,
            environment=settings.ENVIRONMENT,
            database="connected" if db_ok else "unavailable",
            model_loaded=model_ok,
            timestamp=datetime.now(UTC),
        )

    @app.get("/", tags=["system"], summary="API information", include_in_schema=False)
    async def root() -> dict[str, str]:
        """Service name, version and the documentation / health URLs."""
        return {
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "description": settings.APP_DESCRIPTION,
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "api": settings.API_V1_PREFIX,
        }

    # --- routers (versioned prefix; a future /api/v2 can be mounted alongside) ----
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)
    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover - manual entry point
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)  # noqa: S104


__all__ = ["HTTPException", "app", "create_app"]
