"""Optional S3 storage for uploaded audio. Disabled unless S3_BUCKET_NAME is configured."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.config import get_settings

logger = logging.getLogger(__name__)


def s3_enabled() -> bool:
    return bool(get_settings().S3_BUCKET_NAME)


def is_s3_key(path: str | None) -> bool:
    """Object keys produced by build_object_key start with 'audio/'; plain filenames do not."""
    return bool(path) and path.startswith("audio/")


def delete_from_s3(key: str) -> bool:
    """Delete an object. Returns True on success or when S3 is disabled, False on failure."""
    settings = get_settings()
    if not settings.S3_BUCKET_NAME:
        return True
    try:
        import boto3

        client = boto3.client(
            "s3",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID or None,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY or None,
        )
        client.delete_object(Bucket=settings.S3_BUCKET_NAME, Key=key)
        logger.info("Deleted s3://%s/%s", settings.S3_BUCKET_NAME, key)
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
    settings = get_settings()
    if not settings.S3_BUCKET_NAME:
        return None
    key = build_object_key(original_filename or file_path.name, user_id)
    try:
        import boto3

        client = boto3.client(
            "s3",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID or None,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY or None,
        )
        client.upload_file(str(file_path), settings.S3_BUCKET_NAME, key)
        logger.info("Uploaded audio to s3://%s/%s", settings.S3_BUCKET_NAME, key)
        return key
    except Exception:  # noqa: BLE001 - never fail the request because of storage
        logger.exception("S3 upload failed for %s", file_path)
        return None
