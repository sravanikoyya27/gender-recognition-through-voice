"""Audio upload handling: validation, temporary storage, format conversion and loading.

The decoding stack is libsndfile (via soundfile, used by librosa), which reads
WAV/FLAC/OGG and MP3 natively, so no ffmpeg is required.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastapi import UploadFile

from app.config import get_settings

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000
ACCEPTED_MIME_PREFIXES = ("audio/", "video/webm", "application/octet-stream", "application/ogg")


class AudioValidationError(ValueError):
    """Raised when an uploaded file is not acceptable audio (-> HTTP 400)."""


@dataclass(frozen=True)
class DecodedAudio:
    samples: np.ndarray  # mono float32 waveform in [-1, 1] at ``sample_rate``
    sample_rate: int
    native_sample_rate: int
    native_channels: int

    @property
    def duration_seconds(self) -> float:
        return float(len(self.samples)) / float(self.sample_rate) if self.sample_rate else 0.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_audio_file(upload: UploadFile | None) -> str:
    """Check presence, extension and MIME type of an upload. Returns the normalised extension."""
    settings = get_settings()
    if upload is None or not upload.filename:
        raise AudioValidationError("No file provided")

    ext = Path(upload.filename).suffix.lower()
    if ext not in settings.allowed_audio_extensions:
        allowed = ", ".join(sorted(settings.allowed_audio_extensions))
        raise AudioValidationError(f"Invalid audio format '{ext or 'none'}'. Allowed: {allowed}")

    content_type = (upload.content_type or "").lower()
    if content_type and not content_type.startswith(ACCEPTED_MIME_PREFIXES):
        raise AudioValidationError(f"Invalid audio format: unexpected content type '{content_type}'")
    return ext


#: Dedicated directory for upload temp files so shutdown cleanup never touches foreign files.
UPLOAD_TEMP_DIR = Path(tempfile.gettempdir()) / "voice_gender_uploads"


async def save_upload_to_temp(upload: UploadFile, suffix: str) -> Path:
    """Stream the upload to a temporary file, enforcing the size limit while reading."""
    settings = get_settings()
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    written = 0
    UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="upload_", suffix=suffix, dir=str(UPLOAD_TEMP_DIR))
    path = Path(name)
    try:
        with os.fdopen(fd, "wb") as fh:
            while chunk := await upload.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise AudioValidationError(f"File too large: limit is {settings.MAX_UPLOAD_SIZE_MB} MB")
                fh.write(chunk)
    except BaseException:
        cleanup_temp(path)
        raise
    if written == 0:
        cleanup_temp(path)
        raise AudioValidationError("Uploaded file is empty")
    return path


def cleanup_temp(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:  # pragma: no cover - best effort
        logger.warning("Could not remove temporary file %s", path)


def cleanup_temp_files() -> int:
    """Remove every leftover file in the upload temp directory (called at shutdown). Returns the count."""
    if not UPLOAD_TEMP_DIR.exists():
        return 0
    removed = 0
    for path in UPLOAD_TEMP_DIR.iterdir():
        if path.is_file():
            try:
                path.unlink()
                removed += 1
            except OSError:  # pragma: no cover - best effort
                logger.warning("Could not remove temporary file %s", path)
    return removed


# ---------------------------------------------------------------------------
# Conversion and loading
# ---------------------------------------------------------------------------


def convert_audio_format(path: Path, target_suffix: str = ".wav") -> Path:
    """Convert a compressed file (e.g. MP3) to WAV next to it. Returns the new path."""
    import soundfile as sf

    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:  # noqa: BLE001 - libsndfile raises RuntimeError/sf.LibsndfileError
        raise AudioValidationError(f"Corrupted audio file: {exc}") from exc
    converted = path.with_suffix(target_suffix)
    sf.write(str(converted), data, sr, subtype="PCM_16")
    return converted


def load_and_prepare_audio(path: Path) -> DecodedAudio:
    """Load with librosa: mono, resampled to 16 kHz, duration checked (1-10 s by default)."""
    import librosa
    import soundfile as sf

    settings = get_settings()
    try:
        info = sf.info(str(path))
        samples, sr = librosa.load(str(path), sr=TARGET_SAMPLE_RATE, mono=True)
    except Exception as exc:  # noqa: BLE001 - surface any decoder failure uniformly
        raise AudioValidationError(
            f"Corrupted audio file: could not decode ({exc.__class__.__name__})"
        ) from exc

    if samples.size == 0:
        raise AudioValidationError("Corrupted audio file: no samples decoded")

    duration = len(samples) / TARGET_SAMPLE_RATE
    if duration < settings.MIN_AUDIO_SECONDS or duration > settings.MAX_AUDIO_SECONDS:
        raise AudioValidationError(
            f"Audio duration {duration:.2f} s is outside the allowed range "
            f"{settings.MIN_AUDIO_SECONDS:g}-{settings.MAX_AUDIO_SECONDS:g} seconds"
        )
    return DecodedAudio(
        samples=np.asarray(samples, dtype=np.float32),
        sample_rate=TARGET_SAMPLE_RATE,
        native_sample_rate=int(info.samplerate),
        native_channels=int(info.channels),
    )


def decode_audio(data: bytes, target_sr: int | None = TARGET_SAMPLE_RATE) -> DecodedAudio:
    """Decode audio bytes (kept for callers that already hold the bytes in memory)."""
    import io

    import librosa
    import soundfile as sf

    try:
        buffer = io.BytesIO(data)
        info = sf.info(buffer)
        buffer.seek(0)
        samples, sr = librosa.load(buffer, sr=target_sr, mono=True)
    except Exception as exc:  # noqa: BLE001
        raise AudioValidationError(
            f"Corrupted audio file: could not decode ({exc.__class__.__name__})"
        ) from exc
    if samples.size == 0:
        raise AudioValidationError("Corrupted audio file: no samples decoded")
    return DecodedAudio(
        samples=samples.astype(np.float32, copy=False),
        sample_rate=int(sr),
        native_sample_rate=int(info.samplerate),
        native_channels=int(info.channels),
    )
