"""Optional S3 storage for uploaded audio.

Uploads happen only when ``S3_ENABLED=true`` **and** ``S3_BUCKET_NAME`` is set (see
``app.config.Settings``); a bucket name alone, such as the placeholder in an ``.env.example``,
never switches uploads on. Local storage (the original filename recorded on the prediction row,
no copy kept) is the default.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.config import get_settings

logger = logging.getLogger(__name__)


def s3_enabled() -> bool:
    """True only with the explicit switch on and a bucket configured (validated in Settings)."""
    settings = get_settings()
    return bool(settings.S3_ENABLED and settings.S3_BUCKET_NAME)


def is_s3_key(path: str | None) -> bool:
    """Object keys produced by build_object_key start with 'audio/'; plain filenames do not."""
    return bool(path) and path.startswith("audio/")


def _client():
    import boto3

    settings = get_settings()
    return boto3.client(
        "s3",
        region_name=settings.AWS_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID or None,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY or None,
    )


def delete_from_s3(key: str) -> bool:
    """Delete an object. Returns True on success or when S3 is disabled, False on failure."""
    if not s3_enabled():
        return True
    bucket = get_settings().S3_BUCKET_NAME
    try:
        _client().delete_object(Bucket=bucket, Key=key)
        logger.info("Deleted s3://%s/%s", bucket, key)
        return True
    except Exception:  # noqa: BLE001
        logger.exception("S3 delete failed for %s", key)
        return False


def build_object_key(filename: str, user_id: int | None) -> str:
    now = datetime.now(UTC)
    owner = f"user-{user_id}" if user_id is not None else "anonymous"
    suffix = Path(filename).suffix.lower() or ".wav"
    return f"audio/{owner}/{now:%Y/%m/%d}/{now:%H%M%S}-{uuid4().hex[:8]}{suffix}"


def upload_to_s3(file_path: Path, user_id: int | None, original_filename: str | None = None) -> str | None:
    """Upload ``file_path`` to the configured bucket. Returns the object key, or None when disabled.

    Failures are logged and swallowed: a storage problem must not fail the prediction.
    """
    if not s3_enabled():
        return None
    bucket = get_settings().S3_BUCKET_NAME
    key = build_object_key(original_filename or file_path.name, user_id)
    try:
        _client().upload_file(str(file_path), bucket, key)
        logger.info("Uploaded audio to s3://%s/%s", bucket, key)
        return key
    except Exception:  # noqa: BLE001 - never fail the request because of storage
        logger.exception("S3 upload failed for %s", file_path)
        return None
