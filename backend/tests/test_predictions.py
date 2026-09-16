"""Prediction pipeline tests (Phase 3C contract checks and the Phase 5 specification suite).

Sections
--------
1. Phase 3C: validation, pipeline, persistence, batch, mp3, auth (module fixture ``auth_headers``)
2. Phase 5 unit tests: feature extraction, model inference
3. Phase 5 integration tests: prediction endpoint, authentication, analytics
4. Phase 5 database tests
5. Phase 5 performance tests
6. Sample audio fixture (tests/fixtures/sample_audio.wav)

The Phase 5 tests use the specification fixtures from ``conftest.py`` (``test_client``,
``db_session``, ``test_user``, ``valid_token``, ``test_audio_file``). Deeper auth, analytics
and ORM edge cases live in ``test_auth.py``, ``test_analytics.py`` and
``test_database_models.py``.
"""

from __future__ import annotations

import io
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.main import app
from app.ml.features import (
    FEATURE_NAMES,
    FeatureExtractionError,
    extract_all_features,
    extract_features,
    features_to_vector,
)
from app.ml.inference import CLASS_NAMES, InferenceService
from app.models.database import AcousticFeatures, Prediction, User
from app.services import analytics as analytics_service
from tests.conftest import SAMPLE_AUDIO_PATH, TEST_PASSWORD
from tests.fixtures import generate_sample_audio as sample_spec

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


# ===========================================================================
# Phase 5 specification suite
# ===========================================================================


def _upload(client: TestClient, token: str | None, name: str, data: bytes, mime: str = "audio/wav"):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return client.post(PRED, files={"audio": (name, data, mime)}, headers=headers)


def _inference(client: TestClient) -> InferenceService:
    service: InferenceService = app.state.inference
    if not service.is_loaded:
        pytest.skip("Model artifacts are not deployed (backend/app/ml/models)")
    return service


def _decode(data: bytes) -> tuple[np.ndarray, int]:
    samples, sr = sf.read(io.BytesIO(data), dtype="float32")
    return samples, int(sr)


# ---------------------------------------------------------------------------
# 2a. Unit tests: feature extraction
# ---------------------------------------------------------------------------


def test_extract_features_shape(test_audio_file: bytes) -> None:
    samples, sr = _decode(test_audio_file)
    features = extract_features(samples, sr)
    assert list(features) == list(FEATURE_NAMES) and len(features) == 20
    vector = features_to_vector(features)
    assert vector.shape == (1, 20) and vector.dtype.kind == "f"


def test_extract_features_range(test_audio_file: bytes) -> None:
    samples, sr = _decode(test_audio_file)
    f = extract_features(samples, sr)
    assert all(np.isfinite(v) for v in f.values())
    # Frequency-type features are in kHz and bounded by the analysis bands (docs/acoustic_features.md)
    for name in ("meanfreq", "sd", "median", "Q25", "Q75", "IQR", "mode", "centroid"):
        assert 0.0 <= f[name] <= 0.28, name  # spectral properties come from the 0-280 Hz band
    assert f["Q25"] <= f["median"] <= f["Q75"]
    assert f["IQR"] == pytest.approx(f["Q75"] - f["Q25"], abs=1e-9)
    assert 0.0 <= f["sp.ent"] <= 1.0 and 0.0 <= f["sfm"] <= 1.0
    for name in ("meanfun", "minfun", "maxfun"):
        assert 0.0 < f[name] <= 0.28, name  # cepstral fundamental, fmax = 280 Hz
    assert f["minfun"] <= f["meanfun"] <= f["maxfun"]
    for name in ("meandom", "mindom", "maxdom", "dfrange"):
        assert 0.0 <= f[name] <= 8.0, name  # dominant frequency band is clipped at sr/2 (8 kHz)
    assert f["mindom"] <= f["meandom"] <= f["maxdom"]
    assert f["dfrange"] == pytest.approx(f["maxdom"] - f["mindom"], abs=1e-9)
    assert f["modindx"] >= 0.0
    # The known fundamental of the fixture (120 Hz) must be recovered
    assert f["meanfun"] * 1000 == pytest.approx(sample_spec.FUNDAMENTAL_HZ, rel=0.06)


def test_extract_features_invalid_audio() -> None:
    with pytest.raises(FeatureExtractionError):
        extract_features(np.zeros(SR * 2, dtype=np.float32), SR)  # silence: no voiced frames
    with pytest.raises(FeatureExtractionError):
        extract_features(np.zeros(0, dtype=np.float32), SR)  # empty
    with pytest.raises(FeatureExtractionError):
        extract_features(np.full(SR // 100, 0.5, dtype=np.float32), SR)  # 10 ms, shorter than a window
    with pytest.raises((FeatureExtractionError, ValueError)):
        extract_features(np.full(SR * 2, np.nan, dtype=np.float32), SR)  # non-finite input
    with pytest.raises(FeatureExtractionError, match="Missing features"):
        features_to_vector({"meanfreq": 0.1})  # an incomplete feature dict cannot be vectorised


# ---------------------------------------------------------------------------
# 2b. Unit tests: model inference
# ---------------------------------------------------------------------------


def test_model_loads(client: TestClient) -> None:
    settings = get_settings()
    assert settings.MODEL_PATH.exists(), "model_mlp.h5 is missing (run ml_models/model_comparison.py)"
    assert settings.SCALER_PATH.exists()
    assert settings.MODEL_MANIFEST_PATH.exists()
    service = _inference(client)
    info = service.info()
    assert info["loaded"] is True and info["scaler_loaded"] is True
    assert info["model_name"] == "MLP" and info["version"]
    assert service.manifest is not None and service.manifest["input"]["dimension"] == 20
    assert tuple(service.manifest["input"]["feature_order"]) == FEATURE_NAMES
    assert info["load_seconds"] is not None and info["load_seconds"] > 0


def test_model_prediction_shape(client: TestClient, test_audio_file: bytes) -> None:
    service = _inference(client)
    samples, sr = _decode(test_audio_file)
    vector = features_to_vector(extract_features(samples, sr))
    probs = service.predict_proba(vector)
    assert probs.shape == (1, 2)
    batch = service.predict_proba(np.vstack([vector, vector, vector]))
    assert batch.shape == (3, 2)
    with pytest.raises(ValueError):
        service.predict_proba(np.zeros((1, 19)))


def test_model_prediction_range(client: TestClient, test_audio_file: bytes) -> None:
    service = _inference(client)
    samples, sr = _decode(test_audio_file)
    result = service.predict(features_to_vector(extract_features(samples, sr)))
    probs = result["probabilities"]
    assert set(probs) == set(CLASS_NAMES)
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-5)  # softmax output
    assert 0.5 <= result["confidence"] <= 1.0
    assert result["predicted_gender"] == ("male" if probs["male"] > 0.5 else "female")
    assert result["confidence"] == pytest.approx(max(probs.values()))


def test_model_inference_time(client: TestClient, test_audio_file: bytes) -> None:
    """Specification budget: a single prediction completes in under 500 ms."""
    service = _inference(client)
    samples, sr = _decode(test_audio_file)
    vector = features_to_vector(extract_features(samples, sr))
    timings = []
    for _ in range(5):
        start = time.perf_counter()
        service.predict(vector)
        timings.append((time.perf_counter() - start) * 1000)
    assert min(timings) < 500, timings
    assert sum(timings) / len(timings) < 500, timings


# ---------------------------------------------------------------------------
# 3a. Integration tests: prediction endpoint
# ---------------------------------------------------------------------------


def test_predict_with_valid_audio(test_client: TestClient, valid_token: str, test_audio_file: bytes) -> None:
    r = _upload(test_client, valid_token, "sample_audio.wav", test_audio_file)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["gender"] in ("male", "female")
    assert 0.5 <= body["confidence"] <= 1.0
    assert body["audio_duration"] == pytest.approx(sample_spec.DURATION_SECONDS, abs=0.01)
    assert body["sample_rate"] == sample_spec.SAMPLE_RATE
    assert list(body["audio_features"]) == list(FEATURE_NAMES)
    assert body["audio_features"]["meanfun"] * 1000 == pytest.approx(sample_spec.FUNDAMENTAL_HZ, rel=0.06)
    assert body["prediction_id"] is not None and body["model_used"] == "MLP"


def test_predict_with_invalid_file(test_client: TestClient, valid_token: str) -> None:
    r = _upload(test_client, valid_token, "notes.txt", b"this is not audio", "text/plain")
    assert r.status_code == 400
    assert "Invalid audio format" in r.json()["detail"]


def test_predict_unauthenticated(test_client: TestClient, test_audio_file: bytes) -> None:
    r = _upload(test_client, None, "sample_audio.wav", test_audio_file)
    assert r.status_code == 401
    assert r.json()["detail"] == "Not authenticated"
    assert r.headers.get("www-authenticate") == "Bearer"


def test_predict_file_too_large(test_client: TestClient, valid_token: str) -> None:
    eleven_mb = b"RIFF" + b"\0" * (11 * 1024 * 1024)
    r = _upload(test_client, valid_token, "huge.wav", eleven_mb)
    assert r.status_code == 400
    assert "File too large" in r.json()["detail"] and "10 MB" in r.json()["detail"]


def test_predict_audio_too_short(test_client: TestClient, valid_token: str) -> None:
    r = _upload(test_client, valid_token, "blip.wav", wav_bytes(synth_voice(120.0, seconds=0.4)))
    assert r.status_code == 400
    assert "outside the allowed range" in r.json()["detail"]


# ---------------------------------------------------------------------------
# 3b. Integration tests: authentication
# ---------------------------------------------------------------------------


def test_register_success(test_client: TestClient) -> None:
    payload = {
        "username": "phase5user",
        "email": "phase5user@example.com",
        "password": TEST_PASSWORD,
        "password_confirm": TEST_PASSWORD,
    }
    r = test_client.post(f"{AUTH}/register", json=payload)
    assert r.status_code == 201, r.text
    user = r.json()["user"]
    assert isinstance(user["id"], int) and user["id"] > 0
    assert user["username"] == "phase5user" and user["email"] == "phase5user@example.com"
    assert "password" not in user and "password_hash" not in user


def test_register_duplicate_email(test_client: TestClient, test_user: dict) -> None:
    r = test_client.post(
        f"{AUTH}/register",
        json={
            "username": f"{test_user['username']}dup",
            "email": test_user["email"],
            "password": TEST_PASSWORD,
            "password_confirm": TEST_PASSWORD,
        },
    )
    assert r.status_code == 409
    assert r.json()["detail"] == "Email is already registered"


def test_login_success(test_client: TestClient, test_user: dict) -> None:
    credentials = {"email": test_user["email"], "password": test_user["password"]}
    r = test_client.post(f"{AUTH}/login", json=credentials)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["token_type"] == "bearer" and body["access_token"].count(".") == 2
    assert body["expires_in"] > 0 and body["refresh_token"]


def test_login_invalid_password(test_client: TestClient, test_user: dict) -> None:
    r = test_client.post(f"{AUTH}/login", json={"email": test_user["email"], "password": "wrong-password"})
    assert r.status_code == 401
    assert r.json()["detail"] == "Invalid email or password"


def test_protected_route_no_token(test_client: TestClient) -> None:
    for path in (f"{AUTH}/me", "/api/v1/me/predictions", "/api/v1/me/statistics"):
        r = test_client.get(path)
        assert r.status_code == 401, path
        assert r.json()["detail"] == "Not authenticated"


def test_protected_route_with_token(test_client: TestClient, test_user: dict, valid_token: str) -> None:
    headers = {"Authorization": f"Bearer {valid_token}"}
    me = test_client.get(f"{AUTH}/me", headers=headers)
    assert me.status_code == 200 and me.json()["id"] == test_user["id"]
    assert test_client.get("/api/v1/me/predictions", headers=headers).status_code == 200
    assert test_client.get("/api/v1/me/statistics", headers=headers).status_code == 200


# ---------------------------------------------------------------------------
# 3c. Integration tests: analytics
# ---------------------------------------------------------------------------


@pytest.fixture
def user_with_history(test_client: TestClient, valid_token: str, test_audio_file: bytes) -> dict:
    """``test_user`` after three real predictions (the fixture recording plus two synthetic tones)."""
    made = []
    for name, data in (
        ("sample_audio.wav", test_audio_file),
        ("high.wav", wav_bytes(synth_voice(230.0, seconds=2.0, seed=61))),
        ("low.wav", wav_bytes(synth_voice(105.0, seconds=1.5, seed=62))),
    ):
        r = _upload(test_client, valid_token, name, data)
        assert r.status_code == 200, r.text
        made.append(r.json())
    return {"token": valid_token, "predictions": made}


def test_get_predictions_history(test_client: TestClient, user_with_history: dict) -> None:
    headers = {"Authorization": f"Bearer {user_with_history['token']}"}
    r = test_client.get("/api/v1/me/predictions", headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 3 and len(body["predictions"]) == 3
    expected_ids = sorted((p["prediction_id"] for p in user_with_history["predictions"]), reverse=True)
    assert [p["id"] for p in body["predictions"]] == expected_ids  # newest first
    by_id = {p["prediction_id"]: p for p in user_with_history["predictions"]}
    for item in body["predictions"]:
        made = by_id[item["id"]]
        assert item["gender"] == made["gender"]
        assert item["confidence"] == pytest.approx(made["confidence"])
        assert item["audio_duration"] == pytest.approx(made["audio_duration"], abs=1e-3)
        assert item["model_used"] == "MLP" and item["created_at"]


def test_get_statistics(test_client: TestClient, user_with_history: dict) -> None:
    headers = {"Authorization": f"Bearer {user_with_history['token']}"}
    r = test_client.get("/api/v1/me/statistics", headers=headers)
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["total_predictions"] == 3
    assert set(s) >= {
        "gender_distribution",
        "average_confidence",
        "confidence_by_gender",
        "average_audio_duration",
        "duration_range",
        "predictions_by_date",
        "generated_at",
        "cached",
    }
    assert s["gender_distribution"]["male"] + s["gender_distribution"]["female"] == 3
    assert s["predictions_by_date"][0]["count"] == 3
    assert s["cached"] is False
    assert test_client.get("/api/v1/me/statistics", headers=headers).json()["cached"] is True


def test_statistics_accuracy(test_client: TestClient, user_with_history: dict) -> None:
    """The API aggregates must equal values computed independently from the predictions made."""
    made = user_with_history["predictions"]
    headers = {"Authorization": f"Bearer {user_with_history['token']}"}
    s = test_client.get("/api/v1/me/statistics", headers=headers).json()

    confidences = [p["confidence"] for p in made]
    durations = [p["audio_duration"] for p in made]
    males = [p for p in made if p["gender"] == "male"]
    females = [p for p in made if p["gender"] == "female"]

    assert s["average_confidence"] == pytest.approx(np.mean(confidences), abs=1e-4)
    assert s["average_audio_duration"] == pytest.approx(np.mean(durations), abs=1e-3)
    rng = s["duration_range"]
    assert rng["min"] == pytest.approx(min(durations), abs=1e-3)
    assert rng["max"] == pytest.approx(max(durations), abs=1e-3)
    assert rng["mean"] == pytest.approx(np.mean(durations), abs=1e-3)
    assert rng["std"] == pytest.approx(np.std(durations), abs=1e-3)  # population std
    dist = s["gender_distribution"]
    assert (dist["male"], dist["female"]) == (len(males), len(females))
    assert dist["male_percentage"] == pytest.approx(100 * len(males) / 3, abs=0.01)
    assert dist["female_percentage"] == pytest.approx(100 * len(females) / 3, abs=0.01)
    for gender, group in (("male", males), ("female", females)):
        if group:
            expected = np.mean([p["confidence"] for p in group])
            assert s["confidence_by_gender"][gender] == pytest.approx(expected, abs=1e-4)
        else:
            assert s["confidence_by_gender"][gender] is None


# ---------------------------------------------------------------------------
# 4. Database tests
# ---------------------------------------------------------------------------


def test_user_creation(db_session: Session) -> None:
    user = User(username="dbuser", email="dbuser@example.com")
    user.set_password(TEST_PASSWORD)
    db_session.add(user)
    db_session.commit()

    stored = db_session.execute(select(User).where(User.email == "dbuser@example.com")).scalar_one()
    assert stored.id == user.id and stored.username == "dbuser"
    assert stored.password_hash != TEST_PASSWORD and stored.check_password(TEST_PASSWORD)
    assert stored.created_at is not None and stored.updated_at is not None


def test_prediction_creation(db_session: Session, test_user: dict, test_audio_file: bytes) -> None:
    samples, sr = _decode(test_audio_file)
    features = extract_all_features(samples, sr)
    row = Prediction(
        user_id=test_user["id"],
        audio_file_path="sample_audio.wav",
        audio_duration=3.0,
        predicted_gender="male",
        confidence_score=0.91,
        probabilities={"male": 0.91, "female": 0.09},
        model_used="MLP",
        model_version="test",
        inference_time_ms=5.0,
    )
    row.acoustic_features.append(AcousticFeatures.from_features(features))
    db_session.add(row)
    db_session.commit()

    stored = db_session.get(Prediction, row.id)
    assert stored is not None and stored.user_id == test_user["id"] and stored.created_at is not None
    assert len(stored.acoustic_features) == 1
    saved = stored.acoustic_features[0].to_dict()
    assert len(saved) == 22
    for name in FEATURE_NAMES:
        assert saved[name] == pytest.approx(features[name], rel=1e-9)
    assert saved["duration"] == pytest.approx(3.0, abs=0.01)


def test_cascade_delete(
    db_session: Session,
    test_user: dict,
    valid_token: str,
    test_client: TestClient,
    test_audio_file: bytes,
) -> None:
    assert _upload(test_client, valid_token, "sample_audio.wav", test_audio_file).status_code == 200
    user = db_session.get(User, test_user["id"])
    assert user is not None
    prediction_ids = [p.id for p in user.predictions]
    assert prediction_ids
    features_of_user = db_session.query(AcousticFeatures).filter(
        AcousticFeatures.prediction_id.in_(prediction_ids)
    )
    assert features_of_user.count() == len(prediction_ids)

    db_session.delete(user)
    db_session.commit()

    assert db_session.get(User, test_user["id"]) is None
    assert db_session.query(Prediction).filter(Prediction.id.in_(prediction_ids)).count() == 0
    assert features_of_user.count() == 0


# ---------------------------------------------------------------------------
# 5. Performance tests
# ---------------------------------------------------------------------------


def test_api_response_time(test_client: TestClient, valid_token: str, test_audio_file: bytes) -> None:
    """End-to-end upload -> decode -> features -> model -> persist in under 2 seconds."""
    start = time.perf_counter()
    r = _upload(test_client, valid_token, "sample_audio.wav", test_audio_file)
    elapsed = time.perf_counter() - start
    assert r.status_code == 200, r.text
    assert elapsed < 2.0, f"{elapsed:.2f} s"
    assert r.json()["total_time_ms"] < 2000


def test_database_query_time(db_session: Session, user_with_history: dict, test_user: dict) -> None:
    """History listing and statistics queries stay under 100 ms."""
    start = time.perf_counter()
    total, rows = analytics_service.list_predictions(
        db_session, test_user["id"], limit=10, offset=0, sort_by="-date", gender_filter="all"
    )
    listing_ms = (time.perf_counter() - start) * 1000
    assert total == 3 and len(rows) == 3
    assert listing_ms < 100, f"{listing_ms:.1f} ms"

    start = time.perf_counter()
    stats = analytics_service.compute_statistics(db_session, test_user["id"])
    stats_ms = (time.perf_counter() - start) * 1000
    assert stats["total_predictions"] == 3
    assert stats_ms < 100, f"{stats_ms:.1f} ms"


# ---------------------------------------------------------------------------
# 6. Sample audio fixture
# ---------------------------------------------------------------------------


def test_sample_audio_fixture(test_audio_file: bytes) -> None:
    assert SAMPLE_AUDIO_PATH.exists() and SAMPLE_AUDIO_PATH.stat().st_size == len(test_audio_file)
    info = sf.info(io.BytesIO(test_audio_file))
    assert info.samplerate == sample_spec.SAMPLE_RATE and info.channels == 1
    assert info.subtype == "PCM_16" and info.format == "WAV"
    assert info.duration == pytest.approx(sample_spec.DURATION_SECONDS, abs=1e-4)
    samples, _ = _decode(test_audio_file)
    assert np.max(np.abs(samples)) == pytest.approx(sample_spec.PEAK, abs=0.01)
    # The committed file must be exactly what the generator produces (deterministic seed)
    assert np.allclose(samples, sample_spec.synthesize(), atol=1.0 / 32768)
