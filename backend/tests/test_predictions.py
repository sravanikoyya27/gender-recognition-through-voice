"""Prediction API tests: validation, pipeline, persistence, batch, mp3, auth."""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.ml.features import FEATURE_NAMES
from app.models.database import AcousticFeatures, Prediction

PRED = "/api/v1/predictions"
AUTH = "/api/v1/auth"
SR = 16_000


def synth_voice(f0: float, seconds: float = 2.0, sr: int = SR, seed: int = 0) -> np.ndarray:
    """Harmonic series with formant-like weighting: enough structure for the feature extractor."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    formants = ((500.0, 80.0), (1500.0, 120.0), (2500.0, 150.0))
    y = np.zeros_like(t)
    for k in range(1, 30):
        fk = k * f0
        if fk >= sr / 2:
            break
        env = sum(np.exp(-0.5 * ((fk - fc) / bw) ** 2) for fc, bw in formants) + 0.05
        y += env / k * np.sin(2 * np.pi * fk * t)
    y += 0.005 * rng.standard_normal(len(t))
    return (0.8 * y / np.max(np.abs(y))).astype(np.float32)


def wav_bytes(samples: np.ndarray, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def mp3_bytes(samples: np.ndarray, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="MP3")
    return buf.getvalue()


@pytest.fixture(scope="module")
def auth_headers(client: TestClient) -> dict[str, str]:
    client.post(
        f"{AUTH}/register",
        json={
            "username": "predictor",
            "email": "predictor@example.com",
            "password": "s3cretpassword",
            "password_confirm": "s3cretpassword",
        },
    )
    token = client.post(
        f"{AUTH}/login", json={"email": "predictor@example.com", "password": "s3cretpassword"}
    )
    return {"Authorization": f"Bearer {token.json()['access_token']}"}


def _post(
    client: TestClient, headers: dict, name: str, data: bytes, mime: str = "audio/wav", field: str = "audio"
):
    return client.post(PRED, files={field: (name, data, mime)}, headers=headers)


# ---------------------------------------------------------------------------
# Authentication and validation
# ---------------------------------------------------------------------------


def test_requires_authentication(client: TestClient) -> None:
    r = _post(client, {}, "voice.wav", wav_bytes(synth_voice(120.0)))
    assert r.status_code == 401 and r.json()["detail"] == "Not authenticated"


def test_no_file_provided(client: TestClient, auth_headers: dict) -> None:
    r = client.post(PRED, files={"other": ("a.txt", b"x", "text/plain")}, headers=auth_headers)
    assert r.status_code == 400 and r.json()["detail"] == "No file provided"


def test_invalid_format_and_mime(client: TestClient, auth_headers: dict) -> None:
    r = _post(client, auth_headers, "voice.txt", b"abc", "text/plain")
    assert r.status_code == 400 and "Invalid audio format" in r.json()["detail"]
    r = _post(client, auth_headers, "voice.wav", wav_bytes(synth_voice(120.0)), "text/plain")
    assert r.status_code == 400 and "content type" in r.json()["detail"]


def test_file_too_large(client: TestClient, auth_headers: dict) -> None:
    r = _post(client, auth_headers, "big.wav", b"\0" * (10 * 1024 * 1024 + 1))
    assert r.status_code == 400 and "File too large" in r.json()["detail"]


def test_empty_and_corrupted_files(client: TestClient, auth_headers: dict) -> None:
    r = _post(client, auth_headers, "empty.wav", b"")
    assert r.status_code == 400 and "empty" in r.json()["detail"].lower()
    r = _post(client, auth_headers, "bad.wav", b"RIFF not really a wav file")
    assert r.status_code == 400 and "Corrupted audio file" in r.json()["detail"]


def test_duration_limits(client: TestClient, auth_headers: dict) -> None:
    short = _post(client, auth_headers, "short.wav", wav_bytes(synth_voice(120.0, seconds=0.5)))
    assert short.status_code == 400 and "outside the allowed range" in short.json()["detail"]
    long = _post(client, auth_headers, "long.wav", wav_bytes(synth_voice(120.0, seconds=11.0)))
    assert long.status_code == 400 and "outside the allowed range" in long.json()["detail"]


def test_silence_is_unprocessable(client: TestClient, auth_headers: dict) -> None:
    r = _post(client, auth_headers, "silence.wav", wav_bytes(np.zeros(SR * 2, dtype=np.float32)))
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def test_successful_prediction_and_persistence(client: TestClient, auth_headers: dict, test_engine) -> None:
    r = _post(client, auth_headers, "clip.wav", wav_bytes(synth_voice(120.0, seconds=2.5, seed=1)))
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["gender"] in ("male", "female") and body["gender"] == body["predicted_gender"]
    assert 0.5 <= body["confidence"] <= 1.0
    assert body["confidence_percentage"] == pytest.approx(body["confidence"] * 100, abs=0.01)
    assert set(body["probabilities"]) == {"male", "female"}
    assert body["probabilities"][body["gender"]] == pytest.approx(body["confidence"])
    assert body["audio_duration"] == pytest.approx(2.5, abs=0.01) and body["sample_rate"] == 16_000
    assert list(body["audio_features"]) == list(FEATURE_NAMES) and body["features"] == body["audio_features"]
    assert body["model_used"] == "MLP" and body["model_version"]
    assert body["audio_filename"] == "clip.wav"
    assert body["prediction_id"] is not None

    # Performance budget from the specification
    assert body["inference_time_ms"] < 500
    assert body["total_time_ms"] < 2000

    # Persisted rows
    with Session(test_engine) as db:
        row = db.get(Prediction, body["prediction_id"])
        assert row is not None
        assert row.predicted_gender == body["gender"] and row.confidence_score == pytest.approx(
            body["confidence"]
        )
        assert row.model_used == "MLP" and row.audio_duration == pytest.approx(2.5, abs=0.01)
        assert row.user is not None and row.user.username == "predictor"
        features = db.query(AcousticFeatures).filter_by(prediction_id=row.id).one()
        stored = features.to_dict(model_features_only=True)
        for name in FEATURE_NAMES:
            assert stored[name] == pytest.approx(body["audio_features"][name], rel=1e-6)
        assert features.duration == pytest.approx(2.5, abs=0.01) and features.peakf is not None


def test_file_field_alias_and_gender_direction(client: TestClient, auth_headers: dict) -> None:
    low = _post(client, auth_headers, "low.wav", wav_bytes(synth_voice(110.0, seed=2)), field="file")
    high = _post(client, auth_headers, "high.wav", wav_bytes(synth_voice(230.0, seed=3)), field="file")
    assert low.status_code == 200 and high.status_code == 200
    # Plumbing check on synthetic tones, not an accuracy claim
    assert low.json()["probabilities"]["male"] > high.json()["probabilities"]["male"]


def test_mp3_is_converted(client: TestClient, auth_headers: dict) -> None:
    if "MP3" not in sf.available_formats():
        pytest.skip("libsndfile without MP3 support")
    r = _post(client, auth_headers, "clip.mp3", mp3_bytes(synth_voice(150.0, seed=4)), "audio/mpeg")
    assert r.status_code == 200, r.text
    assert r.json()["audio_filename"] == "clip.mp3"
    assert r.json()["audio_duration"] == pytest.approx(2.0, abs=0.1)


def test_stereo_44k_is_downmixed_and_resampled(client: TestClient, auth_headers: dict) -> None:
    mono = synth_voice(130.0, seconds=2.0, sr=44_100, seed=5)
    stereo = np.stack([mono, 0.8 * mono], axis=1)
    r = _post(client, auth_headers, "stereo.wav", wav_bytes(stereo, sr=44_100))
    assert r.status_code == 200, r.text
    assert r.json()["sample_rate"] == 16_000 and r.json()["audio_duration"] == pytest.approx(2.0, abs=0.01)


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


def test_batch_prediction(client: TestClient, auth_headers: dict) -> None:
    files = [
        ("files", ("a.wav", wav_bytes(synth_voice(115.0, seed=6)), "audio/wav")),
        ("files", ("b.wav", wav_bytes(synth_voice(210.0, seed=7)), "audio/wav")),
        ("files", ("c.txt", b"nope", "text/plain")),
    ]
    r = client.post(f"{PRED}/batch", files=files, headers=auth_headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["count"], body["succeeded"], body["failed"]) == (3, 2, 1)
    assert [item["status"] for item in body["results"]] == ["ok", "ok", "error"]
    assert body["results"][0]["result"]["prediction_id"] is not None
    assert "Invalid audio format" in body["results"][2]["error"]


def test_batch_limits(client: TestClient, auth_headers: dict) -> None:
    assert client.post(f"{PRED}/batch", headers=auth_headers).status_code in (400, 422)
    too_many = [("files", (f"{i}.wav", b"x", "audio/wav")) for i in range(11)]
    r = client.post(f"{PRED}/batch", files=too_many, headers=auth_headers)
    assert r.status_code == 400 and "Too many files" in r.json()["detail"]


def test_history_listing_requires_auth(client: TestClient, auth_headers: dict) -> None:
    r = client.get(PRED, headers=auth_headers)
    assert r.status_code == 200 and set(r.json()) == {"items", "total"}
    assert client.get(PRED).status_code == 401
