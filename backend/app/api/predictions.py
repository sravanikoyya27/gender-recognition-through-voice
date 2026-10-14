"""Prediction routes (mounted under the API prefix: ``/api/v1/predictions``).

POST /predictions        one audio file (multipart field ``audio``; ``file`` accepted too)
POST /predictions/batch  several audio files (multipart field ``files``)
GET  /predictions        history in the legacy ``{items, total}`` shape (see also /me/predictions)

Errors: 400 upload/audio problems, 401 not authenticated, 422 no voiced speech,
500 model or database failure, 503 no model loaded.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from app.api.auth import get_current_user, get_optional_user
from app.config import get_settings
from app.ml.features import FEATURE_NAMES, FeatureExtractionError
from app.ml.inference import InferenceService, ModelNotLoadedError
from app.models.database import User, get_db
from app.schemas.prediction import (
    BatchPredictionItem,
    BatchPredictionResponse,
    PredictionListResponse,
    PredictionResponse,
)
from app.services.prediction import InferenceError, PersistenceError, run_batch, run_prediction
from app.utils.audio import (
    AudioValidationError,
    DecodedAudio,
    cleanup_temp,
    convert_audio_format,
    load_and_prepare_audio,
    save_upload_to_temp,
    validate_audio_file,
)

logger = logging.getLogger("app.predictions")

router = APIRouter(prefix="/predictions", tags=["predictions"])


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def prediction_user(
    request: Request,
    strict: User = Depends(get_optional_user),
) -> User | None:
    """Require a JWT when PREDICTIONS_REQUIRE_AUTH is on; otherwise allow anonymous callers."""
    if get_settings().PREDICTIONS_REQUIRE_AUTH and strict is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return strict


def loaded_inference(request: Request) -> InferenceService:
    inference: InferenceService = request.app.state.inference
    if not inference.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No trained model is loaded; run ml_models/model_comparison.py to deploy one",
        )
    return inference


def persistence_session(request: Request, db: Session = Depends(get_db)) -> Session | None:
    """The request session when the database is reachable, else None (prediction still served)."""
    return db if request.app.state.database_available else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _convert_and_decode(path: Path, ext: str) -> tuple[Path, Path | None, DecodedAudio]:
    """CPU-bound part of staging (runs in the thread pool): optional MP3 -> WAV, then decode."""
    converted: Path | None = None
    if ext == ".mp3":
        path = convert_audio_format(path, ".wav")
        converted = path
    return path, converted, load_and_prepare_audio(path)


async def _stage_upload(upload: UploadFile | None) -> tuple[Path, list[Path], DecodedAudio]:
    """Validate, save to a temp file, convert if needed and decode. Returns (path, temp files, audio).

    The upload is streamed on the event loop; decoding and resampling are CPU-bound and run in
    the worker thread pool so that other requests are not stalled meanwhile.
    """
    ext = validate_audio_file(upload)
    assert upload is not None
    temp_files: list[Path] = []
    path = await save_upload_to_temp(upload, ext)
    temp_files.append(path)
    try:
        path, converted, audio = await run_in_threadpool(_convert_and_decode, path, ext)
    except BaseException:
        # A converted file may already exist when decoding fails; make sure the caller cleans it.
        if ext == ".mp3":
            temp_files.append(path.with_suffix(".wav"))
        raise
    if converted is not None:
        temp_files.append(converted)
    return path, temp_files, audio


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=PredictionResponse,
    summary="Predict the speaker's gender from one audio file",
    responses={
        400: {"description": "No file, too large, invalid format, bad duration or corrupted audio"},
        401: {"description": "Not authenticated"},
        422: {"description": "No voiced speech detected"},
        500: {"description": "Model inference or database failure"},
        503: {"description": "Model not loaded"},
    },
)
async def create_prediction(
    audio: UploadFile | None = File(default=None, description="Audio file (.wav or .mp3)"),
    file: UploadFile | None = File(default=None, description="Alias of 'audio'"),
    user: User | None = Depends(prediction_user),
    inference: InferenceService = Depends(loaded_inference),
    db: Session | None = Depends(persistence_session),
) -> PredictionResponse:
    """Classify one recording (.wav or .mp3, up to 10 MB, 1-10 s) and store the result."""
    started = time.perf_counter()
    upload = audio or file
    temp_files: list[Path] = []
    try:
        path, temp_files, decoded = await _stage_upload(upload)
        # Feature extraction, TensorFlow inference, the database commit and the optional S3
        # upload are all synchronous: run them in the thread pool so the event loop stays free.
        # The request-scoped Session is only ever used by one thread at a time (here, then in the
        # dependency teardown), which SQLAlchemy supports.
        result = await run_in_threadpool(
            run_prediction,
            decoded,
            inference,
            db=db,
            user_id=user.id if user else None,
            filename=upload.filename if upload else None,
            temp_path=path,
            started_at=started,
        )
        return PredictionResponse(**result)
    except AudioValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except FeatureExtractionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except (InferenceError, PersistenceError) as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    finally:
        for temp in temp_files:
            cleanup_temp(temp)


@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    summary="Predict several audio files in one request",
    responses={400: {"description": "No files or too many files"}, 401: {"description": "Not authenticated"}},
)
async def create_batch_prediction(
    files: list[UploadFile] = File(default=[], description="Audio files"),
    user: User | None = Depends(prediction_user),
    inference: InferenceService = Depends(loaded_inference),
    db: Session | None = Depends(persistence_session),
) -> BatchPredictionResponse:
    """Classify several recordings; invalid files are reported per item, valid ones share one model call."""
    settings = get_settings()
    started = time.perf_counter()
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided")
    if len(files) > settings.MAX_BATCH_FILES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many files: limit is {settings.MAX_BATCH_FILES} per batch",
        )

    temp_files: list[Path] = []
    staged: list[tuple[str | None, DecodedAudio, Path | None]] = []
    items: list[BatchPredictionItem] = []
    positions: list[int] = []  # index into items for each staged file
    try:
        for upload in files:
            try:
                path, temps, decoded = await _stage_upload(upload)
                temp_files.extend(temps)
                positions.append(len(items))
                staged.append((upload.filename, decoded, path))
                items.append(BatchPredictionItem(filename=upload.filename, status="ok"))
            except AudioValidationError as exc:
                items.append(BatchPredictionItem(filename=upload.filename, status="error", error=str(exc)))

        try:
            outcomes = await run_in_threadpool(
                run_batch, staged, inference, db=db, user_id=user.id if user else None
            )
        except (InferenceError, PersistenceError) as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

        for position, outcome in zip(positions, outcomes, strict=True):
            if outcome.get("status") == "ok":
                items[position].result = PredictionResponse(**outcome["result"])
            else:
                items[position].status = "error"
                items[position].error = outcome.get("error", "unknown error")
    finally:
        for temp in temp_files:
            cleanup_temp(temp)

    succeeded = sum(item.status == "ok" for item in items)
    return BatchPredictionResponse(
        count=len(items),
        succeeded=succeeded,
        failed=len(items) - succeeded,
        total_time_ms=round((time.perf_counter() - started) * 1000, 3),
        results=items,
    )


@router.get("", response_model=PredictionListResponse, summary="My predictions (legacy list shape)")
def list_predictions(
    limit: int = 20,
    offset: int = 0,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PredictionListResponse:
    """Same history as /me/predictions, returned as ``{items, total}`` with full prediction objects."""
    from app.services import analytics

    try:
        total, rows = analytics.list_predictions(
            db, user.id, limit=limit, offset=offset, sort_by="-date", gender_filter="all"
        )
    except analytics.InvalidQueryError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    items = []
    for row in rows:
        detailed = analytics.get_prediction(db, user.id, row.id)
        features = analytics.features_of(detailed) if detailed else {}
        model_features = {name: float(features.get(name) or 0.0) for name in FEATURE_NAMES}
        items.append(
            PredictionResponse(
                prediction_id=row.id,
                gender=row.predicted_gender,
                predicted_gender=row.predicted_gender,
                confidence=row.confidence_score,
                confidence_percentage=round(row.confidence_score * 100, 2),
                probabilities=row.probabilities or {},
                inference_time_ms=row.inference_time_ms or 0.0,
                audio_duration=row.audio_duration or 0.0,
                duration_seconds=row.audio_duration or 0.0,
                sample_rate=16_000,
                audio_filename=row.audio_file_path,
                audio_file_path=row.audio_file_path,
                audio_features=model_features,
                features=model_features,
                model_used=row.model_used,
                model_version=row.model_version,
                user_id=row.user_id,
                created_at=row.created_at,
            )
        )
    return PredictionListResponse(items=items, total=total)
