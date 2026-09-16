"""Generate the deterministic test recording ``tests/fixtures/sample_audio.wav``.

Characteristics (asserted by ``tests/test_predictions.py::test_sample_audio_fixture``):

* 3.000 s, 16 kHz, mono, 16-bit PCM WAV (96 044 bytes)
* synthetic voiced signal: harmonic series with fundamental 120 Hz, formant-like
  spectral weighting at 500 / 1500 / 2500 Hz, a little white noise (seed 2020),
  peak amplitude 0.8

The signal is not a human voice; it only has enough harmonic structure for the
specan feature extractor to find a fundamental. Run this script to regenerate the
file after changing the parameters:

    python tests/fixtures/generate_sample_audio.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000
DURATION_SECONDS = 3.0
FUNDAMENTAL_HZ = 120.0
SEED = 2020
FORMANTS = ((500.0, 80.0), (1500.0, 120.0), (2500.0, 150.0))
PEAK = 0.8
OUTPUT = Path(__file__).with_name("sample_audio.wav")


def synthesize() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    t = np.arange(int(SAMPLE_RATE * DURATION_SECONDS)) / SAMPLE_RATE
    y = np.zeros_like(t)
    for k in range(1, 30):
        fk = k * FUNDAMENTAL_HZ
        if fk >= SAMPLE_RATE / 2:
            break
        envelope = sum(np.exp(-0.5 * ((fk - fc) / bw) ** 2) for fc, bw in FORMANTS) + 0.05
        y += envelope / k * np.sin(2 * np.pi * fk * t)
    y += 0.005 * rng.standard_normal(len(t))
    return (PEAK * y / np.max(np.abs(y))).astype(np.float32)


def main() -> None:
    sf.write(str(OUTPUT), synthesize(), SAMPLE_RATE, subtype="PCM_16")
    info = sf.info(str(OUTPUT))
    print(
        f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes): "
        f"{info.duration:.3f} s, {info.samplerate} Hz, {info.channels} ch"
    )


if __name__ == "__main__":
    main()
