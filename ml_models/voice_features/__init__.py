"""voice_features: acoustic feature extraction shared by training and inference.

Single source of truth for the 20-feature vector consumed by the gender
classifier. Import from here, never re-implement the features elsewhere.
"""

from voice_features.specan import (
    CANONICAL_SAMPLE_RATE,
    EXCLUDED_FROM_MODEL,
    MODEL_FEATURE_NAMES,
    N_MODEL_FEATURES,
    SPECAN_FEATURE_NAMES,
    FeatureExtractionError,
    SpecanConfig,
    compute_specan_features,
    extract_model_features,
    features_to_vector,
    load_audio,
)

__version__ = "0.1.0"

__all__ = [
    "CANONICAL_SAMPLE_RATE",
    "EXCLUDED_FROM_MODEL",
    "MODEL_FEATURE_NAMES",
    "N_MODEL_FEATURES",
    "SPECAN_FEATURE_NAMES",
    "FeatureExtractionError",
    "SpecanConfig",
    "compute_specan_features",
    "extract_model_features",
    "features_to_vector",
    "load_audio",
]
