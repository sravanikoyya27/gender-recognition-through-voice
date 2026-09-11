"""Model loading and inference service.

The trained MLP and its scaler are loaded once at application startup (see
``app.main.lifespan``) and cached on ``app.state.inference``. The deployment
manifest ``model_config.json`` written by ``ml_models/model_comparison.py`` is
read when present to verify artifact hashes, feature order and class mapping.

TensorFlow is imported lazily inside ``load()`` so that importing the app stays fast.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.ml.features import FEATURE_NAMES, N_FEATURES

logger = logging.getLogger(__name__)

CLASS_NAMES: tuple[str, str] = ("male", "female")  # output index order of the trained model
MALE_INDEX, FEMALE_INDEX = 0, 1


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is requested before a model is available."""


class ManifestMismatchError(RuntimeError):
    """The deployed artifacts do not match model_config.json."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class InferenceService:
    """Holds the trained MLP, its feature scaler and the deployment manifest."""

    def __init__(self, model_path: Path, scaler_path: Path | None = None, manifest_path: Path | None = None):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path) if scaler_path else None
        self.manifest_path = (
            Path(manifest_path) if manifest_path else self.model_path.with_name("model_config.json")
        )
        self._model: Any = None
        self._scaler: Any = None
        self.manifest: dict[str, Any] | None = None
        self.version: str | None = None
        self.model_name: str = "MLP"
        self.load_seconds: float | None = None

    # ------------------------------------------------------------------ status
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def info(self) -> dict[str, Any]:
        return {
            "loaded": self.is_loaded,
            "model_name": self.model_name,
            "version": self.version,
            "model_path": str(self.model_path),
            "scaler_loaded": self._scaler is not None,
            "manifest": str(self.manifest_path) if self.manifest else None,
            "load_seconds": self.load_seconds,
        }

    # ------------------------------------------------------------------ loading
    def _read_manifest(self) -> None:
        if not self.manifest_path.exists():
            logger.warning("No model manifest at %s; skipping artifact verification", self.manifest_path)
            return
        manifest = json.loads(self.manifest_path.read_text())
        self.manifest = manifest

        feature_order = manifest.get("input", {}).get("feature_order")
        if feature_order and tuple(feature_order) != tuple(FEATURE_NAMES):
            raise ManifestMismatchError("Manifest feature order differs from app.ml.features.FEATURE_NAMES")
        dim = manifest.get("input", {}).get("dimension")
        if dim and int(dim) != N_FEATURES:
            raise ManifestMismatchError(f"Manifest input dimension {dim} != {N_FEATURES}")
        mapping = manifest.get("output", {}).get("class_mapping")
        if mapping and tuple(mapping[str(i)] for i in range(len(CLASS_NAMES))) != CLASS_NAMES:
            raise ManifestMismatchError(f"Manifest class mapping {mapping} differs from {CLASS_NAMES}")

        artifacts = manifest.get("artifacts", {})
        expected = artifacts.get("model_sha256")
        if expected and self.model_path.exists() and _sha256(self.model_path) != expected:
            raise ManifestMismatchError("Model file hash does not match the manifest")
        expected_scaler = artifacts.get("scaler_sha256")
        if expected_scaler and self.scaler_path and self.scaler_path.exists():
            if _sha256(self.scaler_path) != expected_scaler:
                raise ManifestMismatchError("Scaler file hash does not match the manifest")

        selected = manifest.get("selected_model", {})
        self.version = selected.get("model_version") or self.version
        self.model_name = {"keras_sequential_mlp": "MLP"}.get(
            selected.get("model_type", ""), selected.get("key", "MLP")
        )
        logger.info("Model manifest verified: %s v%s", self.model_name, self.version)

    def load(self) -> bool:
        """Load artifacts if present. Returns True on success, False if the model file is missing."""
        if not self.model_path.exists():
            logger.warning("Model artifact not found at %s; predictions are disabled", self.model_path)
            return False
        start = time.perf_counter()
        self.version = self.model_path.stem
        self._read_manifest()  # raises on mismatch: a wrong artifact must not serve silently

        import tensorflow as tf  # noqa: PLC0415 - deliberate lazy import

        self._model = tf.keras.models.load_model(self.model_path, compile=False)  # inference only

        if self.scaler_path and self.scaler_path.exists():
            with self.scaler_path.open("rb") as fh:
                self._scaler = pickle.load(fh)  # StandardScaler fitted on train.csv (Phase 2B)
        else:
            logger.warning("Scaler artifact not found at %s; raw features will be used", self.scaler_path)

        self._warm_up()
        self.load_seconds = time.perf_counter() - start
        logger.info(
            "Loaded model %s v%s from %s in %.1f s",
            self.model_name,
            self.version,
            self.model_path,
            self.load_seconds,
        )
        return True

    def _warm_up(self) -> None:
        """Trace the graph once so the first real request does not pay the start-up cost."""
        self.predict_proba(np.zeros((1, N_FEATURES), dtype=np.float32))

    def unload(self) -> None:
        self._model = None
        self._scaler = None

    # ------------------------------------------------------------------ inference
    def predict_proba(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Scale and run the model on an (n, 20) matrix. Returns (n, 2) probabilities."""
        if not self.is_loaded:
            raise ModelNotLoadedError("No model is loaded")
        X = np.asarray(feature_matrix, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != N_FEATURES:
            raise ValueError(f"Expected a feature matrix of shape (n, {N_FEATURES}), got {X.shape}")
        if self._scaler is not None:
            X = self._scaler.transform(X)
        # Direct call: ~5 ms per single sample versus ~40 ms for model.predict()
        return np.asarray(self._model(X.astype(np.float32), training=False), dtype=np.float64)

    @staticmethod
    def decide(probs: np.ndarray) -> tuple[str, float]:
        """Specification rule: male if P(male) > 0.5 else female; confidence = max probability."""
        gender = CLASS_NAMES[MALE_INDEX] if probs[MALE_INDEX] > 0.5 else CLASS_NAMES[FEMALE_INDEX]
        return gender, float(max(probs[MALE_INDEX], probs[FEMALE_INDEX]))

    def predict(self, feature_vector: np.ndarray) -> dict[str, Any]:
        """Predict one sample: gender, confidence, per-class probabilities and inference time."""
        start = time.perf_counter()
        probs = self.predict_proba(np.asarray(feature_vector).reshape(1, -1))[0]
        inference_ms = (time.perf_counter() - start) * 1000
        gender, confidence = self.decide(probs)
        return {
            "predicted_gender": gender,
            "confidence": confidence,
            "probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, probs, strict=True)},
            "model_used": self.model_name,
            "model_version": self.version,
            "inference_time_ms": inference_ms,
        }

    def predict_batch(self, feature_matrix: np.ndarray) -> list[dict[str, Any]]:
        """Predict many samples in one model call (batching)."""
        start = time.perf_counter()
        probs = self.predict_proba(feature_matrix)
        per_sample_ms = (time.perf_counter() - start) * 1000 / max(len(probs), 1)
        results = []
        for row in probs:
            gender, confidence = self.decide(row)
            results.append(
                {
                    "predicted_gender": gender,
                    "confidence": confidence,
                    "probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, row, strict=True)},
                    "model_used": self.model_name,
                    "model_version": self.version,
                    "inference_time_ms": per_sample_ms,
                }
            )
        return results
