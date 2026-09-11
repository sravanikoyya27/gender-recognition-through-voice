"""Synthetic voiced signals for testing without a real corpus.

A "voice" is a harmonic series at fundamental ``f0`` shaped by three
formant-like resonances, with slight vibrato, a little noise and fade in/out.
It is not speech, but it exercises every measurement (spectral band statistics,
cepstral fundamental, dominant frequency) with known ground truth for f0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SR = 16_000
FORMANTS = ((500.0, 80.0), (1500.0, 120.0), (2500.0, 150.0))


def synth_voice(
    f0: float,
    duration: float = 2.0,
    sr: int = SR,
    n_harmonics: int = 30,
    noise: float = 0.005,
    vibrato_depth: float = 0.01,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sr * duration)
    t = np.arange(n) / sr
    inst_f0 = f0 * (1.0 + vibrato_depth * np.sin(2 * np.pi * 5.0 * t))
    phase = 2 * np.pi * np.cumsum(inst_f0) / sr

    def envelope(freq: float) -> float:
        return sum(np.exp(-0.5 * ((freq - fc) / bw) ** 2) for fc, bw in FORMANTS) + 0.05

    y = np.zeros(n)
    for k in range(1, n_harmonics + 1):
        fk = k * f0
        if fk >= sr / 2:
            break
        y += (envelope(fk) / k) * np.sin(k * phase)
    y += noise * rng.standard_normal(n)

    fade = int(0.02 * sr)
    ramp = np.linspace(0, 1, fade)
    y[:fade] *= ramp
    y[-fade:] *= ramp[::-1]
    return (0.8 * y / np.max(np.abs(y))).astype(np.float32)


@pytest.fixture(scope="session")
def male_signal() -> np.ndarray:
    return synth_voice(110.0, seed=1)


@pytest.fixture(scope="session")
def female_signal() -> np.ndarray:
    return synth_voice(220.0, seed=2)


@pytest.fixture
def synthetic_splits(tmp_path: Path) -> Path:
    """Two separable Gaussian clusters in the 20-feature space, written as train/val/test CSVs.

    Shared by the MLP and scikit-learn training tests so both train on identical data.
    """
    import pandas as pd

    from voice_features import MODEL_FEATURE_NAMES

    rng = np.random.default_rng(0)
    data_dir = tmp_path / "splits"
    data_dir.mkdir()

    def make(n: int) -> pd.DataFrame:
        rows = []
        for label, centre in (("male", -1.0), ("female", 1.0)):
            X = rng.normal(loc=centre, scale=1.0, size=(n, len(MODEL_FEATURE_NAMES)))
            df = pd.DataFrame(X, columns=list(MODEL_FEATURE_NAMES))
            df["label"] = label
            rows.append(df)
        return pd.concat(rows, ignore_index=True).sample(frac=1.0, random_state=0)

    make(120).to_csv(data_dir / "train.csv", index=False)
    make(20).to_csv(data_dir / "val.csv", index=False)
    make(20).to_csv(data_dir / "test.csv", index=False)
    return data_dir


@pytest.fixture
def audio_dataset(tmp_path: Path) -> dict[str, Path]:
    """Directory with 8 male-like + 8 female-like WAVs, plus invalid files, and a labels CSV."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    rows = ["filename,label"]
    male_f0 = np.linspace(95, 140, 8)
    female_f0 = np.linspace(185, 245, 8)
    for i, f0 in enumerate(male_f0):
        name = f"male_{i:02d}.wav"
        sf.write(audio_dir / name, synth_voice(float(f0), duration=1.5 + 0.2 * i, seed=100 + i), SR)
        rows.append(f"{name},male")
    for i, f0 in enumerate(female_f0):
        name = f"female_{i:02d}.wav"
        sf.write(audio_dir / name, synth_voice(float(f0), duration=1.5 + 0.2 * i, seed=200 + i), SR)
        rows.append(f"{name},Female")  # mixed case on purpose

    # Invalid inputs that must be skipped
    (audio_dir / "corrupt.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEjunkjunkjunk")
    sf.write(audio_dir / "too_short.wav", synth_voice(150.0, duration=0.5, seed=3), SR)
    sf.write(audio_dir / "too_long.wav", synth_voice(150.0, duration=11.0, seed=4), SR)
    sf.write(audio_dir / "unlabelled.wav", synth_voice(150.0, duration=2.0, seed=5), SR)
    sf.write(audio_dir / "resampled_44k.wav", synth_voice(120.0, duration=2.0, sr=44_100, seed=6), 44_100)
    rows += ["corrupt.wav,male", "too_short.wav,male", "too_long.wav,female", "resampled_44k.wav,male"]

    labels = tmp_path / "labels.csv"
    labels.write_text("\n".join(rows) + "\n")
    return {"audio_dir": audio_dir, "labels": labels, "output": tmp_path / "out", "logs": tmp_path / "logs"}
