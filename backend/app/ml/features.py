"""Acoustic feature contract for inference.

The implementation lives in the shared ``voice_features`` package
(``ml_models/voice_features``), installed into this environment as an editable
dependency. Importing it here guarantees that inference uses exactly the code,
parameters and feature ordering that produced the training data.
"""

from __future__ import annotations

import numpy as np
from voice_features import (
    CANONICAL_SAMPLE_RATE,
    MODEL_FEATURE_NAMES,
    N_MODEL_FEATURES,
    SPECAN_FEATURE_NAMES,
    FeatureExtractionError,
    compute_specan_features,
    extract_model_features,
)
from voice_features import features_to_vector as _features_to_vector

#: The 20 model features in voice.csv order (see docs/acoustic_features.md).
FEATURE_NAMES: tuple[str, ...] = MODEL_FEATURE_NAMES
N_FEATURES = N_MODEL_FEATURES

__all__ = [
    "CANONICAL_SAMPLE_RATE",
    "FEATURE_NAMES",
    "N_FEATURES",
    "SPECAN_FEATURE_NAMES",
    "FeatureExtractionError",
    "extract_all_features",
    "extract_features",
    "features_to_vector",
]


def extract_features(samples: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Compute the 20 acoustic model features for a mono waveform (resampled to 16 kHz internally).

    Raises ``FeatureExtractionError`` for silent, too-short or non-voiced input.
    """
    return extract_model_features(samples, sample_rate, strict=True)


def extract_all_features(samples: np.ndarray, sample_rate: int) -> dict[str, float]:
    """All 22 specan measurements (the 20 model features plus ``duration`` and ``peakf``)."""
    features = compute_specan_features(samples, sample_rate)
    bad = [name for name in FEATURE_NAMES if not np.isfinite(features[name])]
    if bad:
        raise FeatureExtractionError(
            f"Non-finite feature values {bad}; the recording may contain no voiced speech"
        )
    return features


def features_to_vector(features: dict[str, float]) -> np.ndarray:
    """Order a feature dict into the model input array of shape (1, 20)."""
    return _features_to_vector(features)
