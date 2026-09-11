"""The backend must expose exactly the shared feature definition used for training."""

import numpy as np
import pytest
from voice_features import MODEL_FEATURE_NAMES

from app.ml.features import (
    FEATURE_NAMES,
    N_FEATURES,
    FeatureExtractionError,
    extract_features,
    features_to_vector,
)


def _harmonic(f0: float, sr: int = 16_000, seconds: float = 1.5) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    y = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 12))
    return (0.8 * y / np.max(np.abs(y))).astype(np.float32)


def test_contract_matches_shared_package() -> None:
    assert FEATURE_NAMES == MODEL_FEATURE_NAMES
    assert N_FEATURES == 20


def test_extract_features_end_to_end() -> None:
    features = extract_features(_harmonic(130.0), 16_000)
    assert list(features) == list(FEATURE_NAMES)
    vector = features_to_vector(features)
    assert vector.shape == (1, 20)
    assert np.all(np.isfinite(vector))
    assert features["meanfun"] * 1000 == pytest.approx(130.0, rel=0.06)


def test_native_rate_is_resampled() -> None:
    at_48k = extract_features(_harmonic(130.0, sr=48_000), 48_000)
    at_16k = extract_features(_harmonic(130.0), 16_000)
    assert at_48k["meanfun"] == pytest.approx(at_16k["meanfun"], rel=0.05)


def test_silence_raises() -> None:
    with pytest.raises(FeatureExtractionError):
        extract_features(np.zeros(16_000 * 2, dtype=np.float32), 16_000)
