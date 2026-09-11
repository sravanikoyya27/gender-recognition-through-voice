"""End-to-end tests for data_preparation.py on a synthetic corpus."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from data_preparation import DataPreparation, create_dataset, extract_20_features, load_audio_files
from tests.conftest import SR, synth_voice
from voice_features import MODEL_FEATURE_NAMES


@pytest.fixture
def prep(audio_dataset: dict[str, Path]) -> DataPreparation:
    return DataPreparation(output_dir=audio_dataset["output"], log_dir=audio_dataset["logs"])


def test_load_audio_files_validates_and_skips(prep: DataPreparation, audio_dataset: dict[str, Path]) -> None:
    files = prep.load_audio_files(audio_dataset["audio_dir"])
    statuses = {f.relative_name: f.status for f in files}
    assert statuses["corrupt.wav"] == "corrupt"
    assert statuses["too_short.wav"] == "too_short"
    assert statuses["too_long.wav"] == "too_long"
    assert statuses["resampled_44k.wav"] == "ok"
    assert sum(f.is_valid for f in files) == 18  # 16 labelled + unlabelled + 44k file
    assert prep.skip_counter == {"corrupt": 1, "too_short": 1, "too_long": 1}
    native = {f.relative_name: f.native_sample_rate for f in files if f.is_valid}
    assert native["resampled_44k.wav"] == 44_100
    assert native["male_00.wav"] == 16_000
    assert (audio_dataset["logs"] / "data_preparation.log").exists()


def test_create_dataset_layout(prep: DataPreparation, audio_dataset: dict[str, Path]) -> None:
    df = prep.create_dataset(audio_dataset["audio_dir"], audio_dataset["labels"])
    # 16 labelled valid files + the labelled 44.1 kHz file; unlabelled.wav is skipped
    assert df.shape == (17, 21)
    assert list(df.columns) == [*MODEL_FEATURE_NAMES, "label"]
    assert set(df["label"]) == {"male", "female"}
    assert df["label"].value_counts().to_dict() == {"male": 9, "female": 8}
    assert not df.isna().any().any()
    assert np.isfinite(df[list(MODEL_FEATURE_NAMES)].to_numpy()).all()
    assert prep.skip_counter["no_label"] == 1

    out = audio_dataset["output"]
    assert (out / "voice_features.csv").exists()
    full = pd.read_csv(out / "voice_features_full.csv")
    assert {"filename", "native_sample_rate", "duration", "peakf"} <= set(full.columns)
    assert (out / "feature_statistics.csv").exists()

    # Synthetic ground truth: male-like files have lower fundamental than female-like
    assert df.loc[df.label == "male", "meanfun"].max() < df.loc[df.label == "female", "meanfun"].min()


def test_batched_writes_produce_same_dataset(prep: DataPreparation, audio_dataset: dict[str, Path]) -> None:
    small = prep.create_dataset(audio_dataset["audio_dir"], audio_dataset["labels"], batch_size=3)
    big = DataPreparation(
        output_dir=audio_dataset["output"] / "b", log_dir=audio_dataset["logs"]
    ).create_dataset(audio_dataset["audio_dir"], audio_dataset["labels"], batch_size=1000)
    pd.testing.assert_frame_equal(small.reset_index(drop=True), big.reset_index(drop=True))


def test_splits_are_stratified_and_saved(prep: DataPreparation, audio_dataset: dict[str, Path]) -> None:
    prep.create_dataset(audio_dataset["audio_dir"], audio_dataset["labels"])
    splits = prep.split_dataset()
    total = sum(len(s) for s in splits.values())
    assert total == 17
    assert len(splits["train"]) == 11  # 70 % of 17, rounded by sklearn
    for name, part in splits.items():
        assert set(part["label"]) == {"male", "female"}, name

    paths = prep.save_splits(audio_dataset["output"])
    for key in ("train", "val", "test", "train_scaled", "scaler", "feature_names", "summary"):
        assert paths[key].exists(), key

    scaler = joblib.load(paths["scaler"])
    train = pd.read_csv(paths["train"])
    scaled = scaler.transform(train[list(MODEL_FEATURE_NAMES)].to_numpy())
    np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-8)
    np.testing.assert_allclose(scaled.std(axis=0), 1.0, atol=1e-8)

    train_scaled = pd.read_csv(paths["train_scaled"])
    np.testing.assert_allclose(train_scaled[list(MODEL_FEATURE_NAMES)].to_numpy(), scaled, atol=1e-6)


def test_extract_features_from_path_array_and_scaler(
    prep: DataPreparation, audio_dataset: dict[str, Path]
) -> None:
    path = audio_dataset["audio_dir"] / "male_00.wav"
    from_path = prep.extract_features(path)
    import soundfile as sf

    y, sr = sf.read(path, dtype="float32")
    from_array = prep.extract_features(y, sr)
    assert from_path.shape == (1, 20)
    np.testing.assert_allclose(from_path, from_array, rtol=1e-5)

    prep.create_dataset(audio_dataset["audio_dir"], audio_dataset["labels"])
    prep.save_splits()
    scaled = prep.extract_features(path)  # uses the fitted scaler automatically
    assert scaled.shape == (1, 20)
    assert not np.allclose(scaled, from_path)

    with pytest.raises(ValueError, match="sample_rate"):
        prep.extract_features(y)


def test_plot_and_summary(prep: DataPreparation, audio_dataset: dict[str, Path]) -> None:
    prep.create_dataset(audio_dataset["audio_dir"], audio_dataset["labels"])
    plot = prep.plot_feature_distributions()
    assert plot.exists() and plot.stat().st_size > 10_000
    summary = prep.summary()
    assert summary["dataset_shape"] == [17, 21]
    assert summary["valid_files"] == 18


def test_function_api(audio_dataset: dict[str, Path]) -> None:
    files = load_audio_files(audio_dataset["audio_dir"], log_dir=audio_dataset["logs"])
    assert len(files) == 21
    vec = extract_20_features(synth_voice(120.0), SR, log_dir=audio_dataset["logs"])
    assert vec.shape == (1, 20)
    df = create_dataset(
        audio_dataset["audio_dir"],
        audio_dataset["labels"],
        output_dir=audio_dataset["output"],
        log_dir=audio_dataset["logs"],
    )
    assert df.shape == (17, 21)


def test_labels_csv_validation(prep: DataPreparation, tmp_path: Path, audio_dataset: dict[str, Path]) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\nx,y\n")
    with pytest.raises(ValueError, match="filename column"):
        prep.create_dataset(audio_dataset["audio_dir"], bad)
