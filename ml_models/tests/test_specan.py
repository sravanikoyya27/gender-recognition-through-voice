"""Tests for the specan-compatible feature implementation."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import SR, synth_voice
from voice_features import (
    EXCLUDED_FROM_MODEL,
    MODEL_FEATURE_NAMES,
    SPECAN_FEATURE_NAMES,
    FeatureExtractionError,
    SpecanConfig,
    compute_specan_features,
    extract_model_features,
    features_to_vector,
)
from voice_features.specan import _frames_r_style, _seewave_specprop

VOICE_CSV_COLUMNS = (
    "meanfreq", "sd", "median", "Q25", "Q75", "IQR", "skew", "kurt", "sp.ent", "sfm",
    "mode", "centroid", "meanfun", "minfun", "maxfun", "meandom", "mindom", "maxdom",
    "dfrange", "modindx",
)  # fmt: skip


def test_model_feature_order_matches_voice_csv() -> None:
    assert MODEL_FEATURE_NAMES == VOICE_CSV_COLUMNS
    assert len(SPECAN_FEATURE_NAMES) == 22
    assert EXCLUDED_FROM_MODEL == ("duration", "peakf")


def test_backend_contract_uses_the_same_definition() -> None:
    pytest.importorskip("app.ml.features", reason="backend package not importable from this venv")
    from app.ml.features import FEATURE_NAMES  # type: ignore[import-not-found]

    assert tuple(FEATURE_NAMES) == MODEL_FEATURE_NAMES


def test_vector_shape_and_finite(male_signal: np.ndarray) -> None:
    features = extract_model_features(male_signal, SR)
    vector = features_to_vector(features)
    assert vector.shape == (1, 20)
    assert vector.dtype == np.float32
    assert np.all(np.isfinite(vector))
    assert list(features) == list(MODEL_FEATURE_NAMES)


def test_all_22_measurements_present(male_signal: np.ndarray) -> None:
    feats = compute_specan_features(male_signal, SR)
    assert tuple(feats) == SPECAN_FEATURE_NAMES
    assert feats["duration"] == pytest.approx(len(male_signal) / SR)
    assert feats["peakf"] > 0  # real value, unlike the hard-coded 0 in the voice.csv script


@pytest.mark.parametrize("f0", [100.0, 120.0, 150.0, 200.0, 240.0])
def test_fundamental_frequency_tracks_f0(f0: float) -> None:
    feats = compute_specan_features(synth_voice(f0, seed=int(f0)), SR)
    assert feats["meanfun"] * 1000 == pytest.approx(f0, rel=0.06)
    # nanmean can differ from identical frame values by one ulp
    assert feats["minfun"] <= feats["meanfun"] + 1e-12
    assert feats["meanfun"] <= feats["maxfun"] + 1e-12


def test_male_like_vs_female_like_ordering(male_signal: np.ndarray, female_signal: np.ndarray) -> None:
    male = compute_specan_features(male_signal, SR)
    female = compute_specan_features(female_signal, SR)
    assert male["meanfun"] < female["meanfun"]
    assert male["meanfreq"] < female["meanfreq"]
    assert male["median"] < female["median"]


def test_historical_identities_and_ranges(male_signal: np.ndarray) -> None:
    f = compute_specan_features(male_signal, SR)
    # specprop computes centroid with the same formula as mean -> identical in voice.csv
    assert f["centroid"] == pytest.approx(f["meanfreq"])
    # Band-limited spectral statistics live in 0-280 Hz (0-0.28 kHz)
    for name in ("meanfreq", "median", "Q25", "Q75", "mode", "centroid"):
        assert 0 <= f[name] <= 0.28
    assert f["Q25"] <= f["median"] <= f["Q75"]
    assert f["IQR"] == pytest.approx(f["Q75"] - f["Q25"])
    assert 0 < f["sp.ent"] <= 1
    assert 0 < f["sfm"] <= 1
    # Dominant frequency is bandpassed to 0-7 kHz at 16 kHz (specan clips bp to ceil(sr/2000)-1)
    assert 0 <= f["mindom"] <= f["meandom"] <= f["maxdom"] <= 7.0
    assert f["dfrange"] == pytest.approx(f["maxdom"] - f["mindom"])
    assert f["modindx"] >= 0
    # The synthetic voice's strongest formant is 500 Hz; the dominant track should sit near it
    assert 0.3 <= f["meandom"] <= 0.8


def test_dominant_bandpass_clipping() -> None:
    assert SpecanConfig(sample_rate=16_000).dominant_bandpass_hz == (0.0, 7000.0)
    assert SpecanConfig(sample_rate=44_100).dominant_bandpass_hz == (0.0, 22000.0)
    assert SpecanConfig(sample_rate=16_000).fundamental_hop == 1024
    assert SpecanConfig(sample_rate=16_000).dominant_hop == 2048


def test_resampling_gives_consistent_features() -> None:
    at_16k = compute_specan_features(synth_voice(130.0, sr=16_000, seed=7), 16_000)
    at_44k = compute_specan_features(synth_voice(130.0, sr=44_100, seed=7), 44_100)
    assert at_44k["meanfun"] == pytest.approx(at_16k["meanfun"], rel=0.05)
    assert at_44k["meanfreq"] == pytest.approx(at_16k["meanfreq"], rel=0.10)
    assert at_44k["meandom"] == pytest.approx(at_16k["meandom"], abs=0.1)


def test_deterministic(male_signal: np.ndarray) -> None:
    a = features_to_vector(extract_model_features(male_signal, SR))
    b = features_to_vector(extract_model_features(male_signal, SR))
    np.testing.assert_array_equal(a, b)


def test_stereo_input_is_downmixed(male_signal: np.ndarray) -> None:
    stereo = np.stack([male_signal, male_signal])
    mono = extract_model_features(male_signal, SR)
    both = extract_model_features(stereo, SR)
    assert both["meanfun"] == pytest.approx(mono["meanfun"], rel=1e-6)


@pytest.mark.parametrize(
    "signal, message",
    [
        (np.zeros(SR * 2, dtype=np.float32), "silent"),
        (np.zeros(0, dtype=np.float32), "Empty"),
        (synth_voice(120.0, duration=0.1), "too short"),
        (np.full(SR * 2, np.nan, dtype=np.float32), "NaN"),
    ],
)
def test_invalid_signals_raise(signal: np.ndarray, message: str) -> None:
    with pytest.raises(FeatureExtractionError, match=message):
        compute_specan_features(signal, SR)


def test_white_noise_has_no_fundamental_and_strict_mode_raises() -> None:
    rng = np.random.default_rng(0)
    noise = (0.5 * rng.standard_normal(SR * 2)).astype(np.float32)
    feats = compute_specan_features(noise, SR)
    # Noise still yields finite spectral statistics
    assert np.isfinite(feats["meanfreq"]) and np.isfinite(feats["meandom"])
    # but the cepstral tracker may or may not lock on; strict extraction must be consistent
    model = extract_model_features(noise, SR, strict=False)
    if any(not np.isfinite(v) for v in model.values()):
        with pytest.raises(FeatureExtractionError, match="Non-finite"):
            extract_model_features(noise, SR, strict=True)
    else:
        assert features_to_vector(model).shape == (1, 20)


def test_features_to_vector_requires_all_names() -> None:
    with pytest.raises(FeatureExtractionError, match="Missing features"):
        features_to_vector({"meanfreq": 0.1})


def test_frame_count_matches_r_seq() -> None:
    # R: length(seq(1, n - wl, by = hop))
    def r_count(n: int, wl: int, hop: int) -> int:
        return len(range(1, n - wl + 1, hop))

    for n, wl, hop in [(16, 8, 4), (17, 8, 4), (20, 8, 4), (4097, 2048, 1024), (32000, 2048, 2048)]:
        frames = _frames_r_style(np.ones(n), wl, hop)
        assert frames.shape == (wl, r_count(n, wl, hop)), (n, wl, hop)


def test_specprop_quantile_convention_matches_r() -> None:
    # Uniform 4-bin spectrum: cumulative = .25 .5 .75 1.0
    freq = np.array([0.0, 1.0, 2.0, 3.0])
    amp = np.array([1.0, 1.0, 1.0, 1.0])
    prop = _seewave_specprop(freq, amp, (0.0, 3.0))
    # R: freq[length(cumamp[cumamp <= 0.5]) + 1] -> freq[3] (1-based) = 2.0
    assert prop["median"] == 2.0
    assert prop["Q25"] == 1.0
    assert prop["Q75"] == 3.0
    assert prop["mean"] == pytest.approx(1.5)
    assert prop["cent"] == pytest.approx(prop["mean"])
    assert prop["sfm"] == pytest.approx(1.0)  # flat spectrum
    assert prop["sh"] == pytest.approx(1.0)  # maximal entropy
