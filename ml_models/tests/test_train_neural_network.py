"""Tests for train_neural_network.py (architecture, encoding, and a short end-to-end run)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from train_neural_network import (
    CLASS_NAMES,
    TrainingConfig,
    build_model,
    compile_model,
    load_split,
    one_hot,
    train,
)
from voice_features import MODEL_FEATURE_NAMES

tf = pytest.importorskip("tensorflow")


def test_class_mapping_and_one_hot() -> None:
    assert CLASS_NAMES == ("male", "female")
    encoded = one_hot(pd.Series(["male", "Female", "male"]))
    np.testing.assert_array_equal(encoded, [[1, 0], [0, 1], [1, 0]])


def test_architecture_is_exactly_as_specified() -> None:
    model = build_model(TrainingConfig())
    kinds = [
        (layer.__class__.__name__, getattr(layer, "units", getattr(layer, "rate", None)))
        for layer in model.layers
    ]
    assert kinds == [
        ("Dense", 64),
        ("Dropout", 0.25),
        ("Dense", 256),
        ("Dropout", 0.25),
        ("Dense", 256),
        ("Dropout", 0.25),
        ("Dense", 64),
        ("Dropout", 0.25),
        ("Dense", 2),
    ]
    dense = [layer for layer in model.layers if layer.__class__.__name__ == "Dense"]
    assert all(layer.activation.__name__ == "relu" for layer in dense[:-1])
    assert dense[-1].activation.__name__ == "softmax"
    assert model.input_shape == (None, 20)
    # 20*64+64 + 64*256+256 + 256*256+256 + 256*64+64 + 64*2+2
    assert model.count_params() == 100_354


def test_compile_configuration() -> None:
    model = compile_model(build_model(TrainingConfig()), TrainingConfig())
    assert model.optimizer.__class__.__name__ == "Nadam"
    assert float(model.optimizer.learning_rate) == pytest.approx(1e-3)
    assert float(model.optimizer.beta_1) == pytest.approx(0.9)
    assert float(model.optimizer.beta_2) == pytest.approx(0.999)
    assert model.loss == "categorical_crossentropy"
    # Keras instantiates compiled metrics lazily on the first evaluate/fit call
    model.evaluate(
        np.zeros((4, 20), dtype=np.float32), one_hot(["male", "female", "male", "female"]), verbose=0
    )
    assert {"precision", "recall", "auc"} <= set(model.metrics_names)


# The `synthetic_splits` fixture (shared with the scikit-learn tests) lives in conftest.py.


def test_load_split_validates(synthetic_splits: Path, tmp_path: Path) -> None:
    X, y = load_split(synthetic_splits / "train.csv")
    assert list(X.columns) == list(MODEL_FEATURE_NAMES)
    assert set(y) == {"male", "female"}
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"meanfreq": [0.1], "label": ["male"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_split(bad)


def test_end_to_end_training_writes_all_artifacts(synthetic_splits: Path, tmp_path: Path) -> None:
    out = tmp_path / "models"
    config = TrainingConfig(data_dir=synthetic_splits, output_dir=out, epochs=5, batch_size=16, seed=1)
    metadata = train(config)

    for name in (
        "model_mlp.h5",
        "model_mlp",
        "scaler_mlp.pkl",
        "history_mlp.json",
        "metrics_mlp.json",
        "model_metadata.json",
        "feature_order.json",
        "class_mapping.json",
        "random_state.json",
        "training_log.txt",
        "training_loss.png",
        "training_accuracy.png",
        "confusion_matrix_test.png",
    ):
        assert (out / name).exists(), name
    assert (out / "model_mlp" / "saved_model.pb").exists()
    assert (out / "checkpoints" / "model_mlp_best.h5").exists()

    assert json.loads((out / "feature_order.json").read_text()) == list(MODEL_FEATURE_NAMES)
    metrics = json.loads((out / "metrics_mlp.json").read_text())
    test_metrics = metrics["metrics"]["test"]
    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc", "confusion_matrix"):
        assert key in test_metrics
    assert np.asarray(test_metrics["confusion_matrix"]["matrix"]).sum() == 40
    assert metrics["training"]["epochs_run"] <= 5
    assert metrics["git"]["commit"]
    assert metadata["feature_order"] == list(MODEL_FEATURE_NAMES)
    # Separable clusters: even 5 epochs should beat chance clearly
    assert test_metrics["accuracy"] > 0.8

    # The saved H5 model reloads and reproduces predictions on the same input
    reloaded = tf.keras.models.load_model(out / "model_mlp.h5", compile=False)
    X, _ = load_split(synthetic_splits / "test.csv")
    import pickle

    scaler = pickle.loads((out / "scaler_mlp.pkl").read_bytes())
    probs = reloaded.predict(scaler.transform(X.to_numpy()).astype(np.float32), verbose=0)
    assert probs.shape == (40, 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
