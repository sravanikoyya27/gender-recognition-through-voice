"""S3 storage must stay off unless explicitly enabled with a real bucket (Phase 6A)."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from app.config import Settings, get_settings
from app.services import storage


@pytest.fixture
def fresh_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    """Reload settings from the environment for one test and restore the cached instance after."""
    for name in ("S3_ENABLED", "S3_BUCKET_NAME", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    try:
        yield monkeypatch
    finally:
        get_settings.cache_clear()


def test_s3_is_disabled_by_default(fresh_settings: pytest.MonkeyPatch) -> None:
    assert get_settings().S3_ENABLED is False
    assert get_settings().S3_BUCKET_NAME is None
    assert storage.s3_enabled() is False


def test_bucket_name_alone_does_not_enable_uploads(
    fresh_settings: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fresh_settings.setenv("S3_BUCKET_NAME", "gender-voice-bucket")
    get_settings.cache_clear()
    assert get_settings().S3_BUCKET_NAME == "gender-voice-bucket"
    assert storage.s3_enabled() is False

    # Neither upload nor delete may touch boto3 while disabled.
    fresh_settings.setitem(sys.modules, "boto3", None)  # importing boto3 would raise ImportError
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF")
    assert storage.upload_to_s3(audio, user_id=1, original_filename="clip.wav") is None
    assert storage.delete_from_s3("audio/user-1/2026/01/01/x.wav") is True


def test_explicit_switch_with_real_bucket_enables_uploads(fresh_settings: pytest.MonkeyPatch) -> None:
    fresh_settings.setenv("S3_ENABLED", "true")
    fresh_settings.setenv("S3_BUCKET_NAME", "my-real-voice-uploads")
    get_settings.cache_clear()
    assert storage.s3_enabled() is True
    assert get_settings().s3_storage_enabled is True


@pytest.mark.parametrize(
    "env",
    [
        {"S3_ENABLED": "true"},  # no bucket
        {"S3_ENABLED": "true", "S3_BUCKET_NAME": "gender-voice-bucket"},  # example bucket
        {"S3_ENABLED": "true", "S3_BUCKET_NAME": "real-bucket", "AWS_ACCESS_KEY_ID": "your_access_key"},
        {"S3_ENABLED": "true", "S3_BUCKET_NAME": "real-bucket", "AWS_SECRET_ACCESS_KEY": "your_secret_key"},
    ],
)
def test_enabling_s3_with_placeholders_is_rejected(env: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="S3"):
        Settings(**env)


def test_blank_values_count_as_unset() -> None:
    settings = Settings(S3_BUCKET_NAME="", AWS_ACCESS_KEY_ID="  ", AWS_SECRET_ACCESS_KEY="")
    assert settings.S3_BUCKET_NAME is None
    assert settings.AWS_ACCESS_KEY_ID is None and settings.AWS_SECRET_ACCESS_KEY is None
    assert settings.s3_storage_enabled is False


def test_predictions_store_local_filename_when_s3_is_off(
    client, auth_headers_for_user: dict, test_audio_file: bytes
) -> None:
    r = client.post(
        "/api/v1/predictions",
        files={"audio": ("local-only.wav", test_audio_file, "audio/wav")},
        headers=auth_headers_for_user,
    )
    assert r.status_code == 200, r.text
    assert r.json()["audio_file_path"] == "local-only.wav"  # original filename, no S3 key
    assert not storage.is_s3_key(r.json()["audio_file_path"])
