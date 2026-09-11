"""Python reproduction of the acoustic measurements behind the `voice.csv` dataset.

Historical pipeline (see docs/acoustic_features.md for the full derivation):

* The B.Tech project extracted features in R with ``warbleR::specan()``.
* The public training set ``voice.csv`` (3 168 rows) that the project's Keras MLP
  was trained on was produced by Kory Becker's ``specan3()``, a lightly modified
  copy of ``warbleR::specan`` built on ``seewave``:

  - ``seewave::spec``     -> whole-signal magnitude spectrum (Hann window, max-normalised)
  - ``seewave::specprop`` -> distribution statistics on the 0-280 Hz band
  - ``seewave::fund``     -> cepstral fundamental-frequency track (wl=2048, 50 % overlap)
  - ``seewave::dfreq``    -> dominant-frequency track (wl=2048, 0 % overlap, bandpass)
  - modulation index computed from the dominant-frequency track

This module reimplements each step with numpy / librosa, keeping the same
formulas, parameters and edge-case behaviour (NA handling, index conventions).
All frequency outputs are in kHz to match the dataset.

Naming: functions prefixed ``_seewave_`` mirror a seewave function one-to-one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal.windows import hann

# ---------------------------------------------------------------------------
# Feature names and ordering (the ordering is part of the model contract)
# ---------------------------------------------------------------------------

#: The 22 measurements in the order returned by warbleR::specan / specan3.
SPECAN_FEATURE_NAMES: tuple[str, ...] = (
    "duration",
    "meanfreq",
    "sd",
    "median",
    "Q25",
    "Q75",
    "IQR",
    "skew",
    "kurt",
    "sp.ent",
    "sfm",
    "mode",
    "centroid",
    "peakf",
    "meanfun",
    "minfun",
    "maxfun",
    "meandom",
    "mindom",
    "maxdom",
    "dfrange",
    "modindx",
)

#: Measurements that exist in specan output but are NOT model inputs.
#: ``duration`` is a recording-length artefact and ``peakf`` was hard-coded to 0
#: in the script that produced voice.csv; both columns are absent from voice.csv.
EXCLUDED_FROM_MODEL: tuple[str, ...] = ("duration", "peakf")

#: The 20 model features in voice.csv column order. This must stay identical to
#: backend/app/ml/features.py::FEATURE_NAMES (the backend imports it from here).
MODEL_FEATURE_NAMES: tuple[str, ...] = tuple(
    name for name in SPECAN_FEATURE_NAMES if name not in EXCLUDED_FROM_MODEL
)
N_MODEL_FEATURES = len(MODEL_FEATURE_NAMES)
assert N_MODEL_FEATURES == 20

#: Sample rate every signal is analysed at. Inputs at other rates are resampled.
CANONICAL_SAMPLE_RATE = 16_000


class FeatureExtractionError(ValueError):
    """Raised when a signal cannot yield a valid feature vector."""


@dataclass(frozen=True)
class SpecanConfig:
    """Analysis parameters. Defaults reproduce Becker's specan3 call used for voice.csv."""

    sample_rate: int = CANONICAL_SAMPLE_RATE
    #: FFT window length for the fundamental / dominant frequency tracks (samples).
    window_length: int = 2048
    #: Amplitude threshold (% of max |x|) below which samples are silenced (seewave::afilter).
    threshold_percent: float = 5.0
    #: Frequency band, in Hz, used for the spectral distribution statistics (specprop flim).
    spectral_band_hz: tuple[float, float] = (0.0, 280.0)
    #: Upper bound of the fundamental frequency search (seewave::fund fmax), Hz.
    fundamental_max_hz: float = 280.0
    #: Frame overlap for the fundamental track (seewave::fund ovlp), percent.
    fundamental_overlap_percent: float = 50.0
    #: Frame overlap for the dominant track (seewave::dfreq ovlp), percent.
    dominant_overlap_percent: float = 0.0
    #: Requested bandpass for the dominant track, kHz (specan ``bp``); clipped like specan does.
    dominant_bandpass_khz: tuple[float, float] = (0.0, 22.0)

    @property
    def dominant_bandpass_hz(self) -> tuple[float, float]:
        """specan clips bp[2] to ``ceiling(sr/2000) - 1`` kHz (7 kHz at 16 kHz)."""
        low, high = self.dominant_bandpass_khz
        limit = math.ceil(self.sample_rate / 2000) - 1
        high = min(high, limit)
        return (low * 1000.0, high * 1000.0)

    @property
    def fundamental_hop(self) -> int:
        return int(self.window_length - self.fundamental_overlap_percent * self.window_length / 100)

    @property
    def dominant_hop(self) -> int:
        return int(self.window_length - self.dominant_overlap_percent * self.window_length / 100)


DEFAULT_CONFIG = SpecanConfig()


# ---------------------------------------------------------------------------
# Loading / resampling
# ---------------------------------------------------------------------------


def load_audio(path: str | Path, sample_rate: int = CANONICAL_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float32 at ``sample_rate`` (librosa.load)."""
    import librosa  # slow import, keep local

    samples, sr = librosa.load(str(path), sr=sample_rate, mono=True)
    return samples.astype(np.float32, copy=False), int(sr)


def _as_mono_float(samples: np.ndarray) -> np.ndarray:
    y = np.asarray(samples, dtype=np.float64)
    if y.ndim == 2:  # (channels, n) or (n, channels)
        y = y.mean(axis=0) if y.shape[0] <= 8 else y.mean(axis=1)
    if y.ndim != 1:
        raise FeatureExtractionError(f"Expected a 1-D mono signal, got shape {y.shape}")
    return y


def _resample_if_needed(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return y
    import librosa

    return librosa.resample(y, orig_sr=sr, target_sr=target_sr)


# ---------------------------------------------------------------------------
# seewave building blocks
# ---------------------------------------------------------------------------


def _seewave_afilter(y: np.ndarray, threshold_percent: float) -> np.ndarray:
    """seewave::afilter - zero every sample whose |x| <= threshold % of max |x|."""
    out = y.copy()
    peak = np.max(np.abs(out))
    if peak == 0:
        return out
    out[np.abs(out) <= threshold_percent / 100.0 * peak] = 0.0
    return out


def _hann_symmetric(n: int) -> np.ndarray:
    """seewave::ftwindow(n, "hanning") is the symmetric Hann window."""
    return hann(n, sym=True)


def _frames_r_style(y: np.ndarray, wl: int, hop: int) -> np.ndarray:
    """Frame like R's ``seq(1, n - wl, by = hop)``: returns (wl, N).

    R's sequence never includes a start > n - wl (1-based), which is one frame
    fewer than the usual ``1 + (n - wl) // hop`` when ``n - wl`` is a multiple
    of ``hop``. Framing ``y[: n - 1]`` reproduces R's count and start positions.
    """
    import librosa

    n = len(y)
    if n - 1 < wl:
        raise FeatureExtractionError(f"Signal too short for wl={wl}: {n} samples")
    return librosa.util.frame(y[: n - 1], frame_length=wl, hop_length=hop)


def _seewave_spec(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """seewave::spec(wave, f, plot=FALSE) with default norm=TRUE and at=NULL.

    Single FFT of the whole (Hann-windowed) signal, magnitude of the first n//2
    bins (Nyquist dropped), normalised to a maximum of 1.
    Implemented as a one-frame librosa.stft with n_fft = len(y).
    Returns (freqs_hz, magnitude).
    """
    import librosa

    n = len(y)
    if n < 4:
        raise FeatureExtractionError("Signal too short for a spectrum")
    window = _hann_symmetric(n)
    stft = librosa.stft(y, n_fft=n, hop_length=n, win_length=n, window=window, center=False)
    mag = np.abs(stft[:, 0]).astype(np.float64)[: n // 2]
    peak = mag.max()
    if not np.isfinite(peak) or peak <= 0:
        raise FeatureExtractionError("Signal has no spectral energy (silent input)")
    mag = mag / peak
    freqs_hz = np.arange(n // 2, dtype=np.float64) * sr / n
    return freqs_hz, mag


def _seewave_sfm(amp: np.ndarray) -> float:
    """seewave::sfm - spectral flatness = geometric mean / arithmetic mean (zeros -> 1e-5)."""
    if amp.sum() == 0:
        return float("nan")
    a = np.where(amp == 0, 1e-5, amp)
    geo = float(np.exp(np.mean(np.log(a))))
    ari = float(np.mean(a))
    return geo / ari


def _seewave_sh(amp: np.ndarray) -> float:
    """seewave::sh - Shannon entropy normalised to [0, 1] (zeros -> 1e-7)."""
    a = np.where(amp == 0, 1e-7, amp)
    a = a / a.sum()
    return float(-np.sum(a * np.log(a)) / np.log(len(a)))


def _seewave_specprop(
    freqs_hz: np.ndarray, mag: np.ndarray, flim_hz: tuple[float, float]
) -> dict[str, float]:
    """seewave::specprop(spec, f, flim) - all outputs in Hz (converted to kHz by the caller).

    Note the historical quirks that are reproduced on purpose:
    * ``centroid`` uses the same formula as ``mean`` (they are identical in voice.csv).
    * ``skewness`` / ``kurtosis`` are moments of the *amplitude values* (treating the
      spectrum bins as a sample), not of the frequency distribution.
    """
    sel = (freqs_hz >= flim_hz[0]) & (freqs_hz <= flim_hz[1])
    freq = freqs_hz[sel]
    amp = mag[sel]
    L = len(amp)
    if L < 3:
        raise FeatureExtractionError("Too few spectral bins in the analysis band")
    total = amp.sum()
    if total <= 0:
        raise FeatureExtractionError("No spectral energy inside the analysis band")
    amp = amp / total  # probability mass function over frequency bins
    cumamp = np.cumsum(amp)

    mean = float(np.sum(amp * freq))
    sd = float(np.sqrt(np.sum(amp * (freq - mean) ** 2)))

    def quantile(q: float) -> float:
        # R: freq[length(cumamp[cumamp <= q]) + 1]  (1-based)  ->  freq[count] (0-based)
        idx = int(np.sum(cumamp <= q))
        return float(freq[min(idx, L - 1)])

    median = quantile(0.5)
    q25 = quantile(0.25)
    q75 = quantile(0.75)
    mode = float(freq[int(np.argmax(amp))])
    cent = float(np.sum(freq * amp))

    z = amp - amp.mean()
    w = float(np.std(amp, ddof=1))  # R's sd()
    if w == 0:
        skew = kurt = float("nan")
    else:
        skew = float((np.sum(z**3) / (L - 1)) / w**3)
        kurt = float((np.sum(z**4) / (L - 1)) / w**4)

    return {
        "mean": mean,
        "sd": sd,
        "median": median,
        "Q25": q25,
        "Q75": q75,
        "IQR": q75 - q25,
        "skewness": skew,
        "kurtosis": kurt,
        "sh": _seewave_sh(amp),
        "sfm": _seewave_sfm(amp),
        "mode": mode,
        "cent": cent,
    }


def _seewave_fund(y: np.ndarray, sr: int, config: SpecanConfig) -> np.ndarray:
    """seewave::fund(wave, f, ovlp=50, threshold, fmax, wl) - fundamental track in kHz.

    Per frame (no window): cepstrum = Re(IFFT(log|FFT(frame)|)); the peak is
    searched at quefrencies above ``f // fmax``; a peak at the very first allowed
    quefrency is treated as "no fundamental" (NaN), exactly like the R code.
    """
    wl = config.window_length
    w = _seewave_afilter(y, config.threshold_percent)
    w = np.where(w == 0, 1e-6, w)  # R: avoid log(0)
    frames = _frames_r_style(w, wl, config.fundamental_hop)  # (wl, N)

    spectrum = np.fft.fft(frames, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cepstrum = np.real(np.fft.ifft(np.log(np.abs(spectrum)), axis=0))
    cepstrum = cepstrum[: wl // 2, :]
    cepstrum = np.where(np.isfinite(cepstrum), cepstrum, 0.0)

    fmaxi = int(sr // config.fundamental_max_hz)  # R: f %/% fmax
    search = cepstrum[fmaxi:, :]  # R: z[-c(1:fmaxi), k]
    if search.shape[0] < 2:
        raise FeatureExtractionError("Window too short for the fundamental search range")
    tfund = np.argmax(search, axis=0)  # 0-based within the search range
    fund_hz = np.where(tfund == 0, np.nan, sr / (tfund + fmaxi))  # R: f / (tfund + fmaxi - 1), 1-based
    return fund_hz / 1000.0


def _seewave_dfreq(y: np.ndarray, sr: int, config: SpecanConfig) -> np.ndarray:
    """seewave::dfreq(wave, f, wl, ovlp=0, threshold, bandpass) - dominant track in kHz.

    STFT (Hann window) of the amplitude-filtered signal; bins outside the
    bandpass are zeroed; the arg-max bin per frame gives the dominant frequency.
    A maximum at the first bin (silent frame / DC) yields NaN, like the R code.
    """
    import librosa

    wl = config.window_length
    hop = config.dominant_hop
    w = _seewave_afilter(y, config.threshold_percent)
    n = len(w)
    if n - 1 < wl:
        raise FeatureExtractionError(f"Signal too short for wl={wl}: {n} samples")

    stft = librosa.stft(
        w[: n - 1],  # match R's frame count, see _frames_r_style
        n_fft=wl,
        hop_length=hop,
        win_length=wl,
        window=_hann_symmetric(wl),
        center=False,
    )
    mag = np.abs(stft).astype(np.float64)[: wl // 2, :]  # seewave keeps wl/2 bins

    low_hz, high_hz = config.dominant_bandpass_hz
    lowlimit = int(round(wl * low_hz / sr))  # 1-based bin indices in R
    upperlimit = int(round(wl * high_hz / sr))
    keep_from = max(lowlimit, 1) - 1  # R ignores index 0; convert to 0-based
    keep_to = min(upperlimit, mag.shape[0])  # exclusive
    mask = np.zeros(mag.shape[0], dtype=bool)
    mask[keep_from:keep_to] = True
    mag[~mask, :] = 0.0

    idx = np.argmax(mag, axis=0)  # 0-based; R's y2 == idx + 1
    dom_hz = np.where(idx == 0, np.nan, idx * sr / wl)  # R: (y2 - 1) * f / wl
    return dom_hz / 1000.0


def _modulation_index(dom_khz: np.ndarray, mindom: float, maxdom: float, dfrange: float) -> float:
    """specan: mean |y[j] - y[j+1]| over non-NA j, divided by dfrange (0 if flat)."""
    if not np.isfinite(mindom) or not np.isfinite(maxdom) or mindom == maxdom:
        return 0.0
    valid = np.flatnonzero(~np.isnan(dom_khz))
    changes = []
    for j in valid:
        if j + 1 < len(dom_khz):
            changes.append(abs(dom_khz[j] - dom_khz[j + 1]))  # NaN when y[j+1] is NA
    changes_arr = np.asarray(changes, dtype=np.float64)
    if changes_arr.size == 0 or np.all(np.isnan(changes_arr)):
        return float("nan")
    return float(np.nanmean(changes_arr) / dfrange)


def _nan_stats(track_khz: np.ndarray) -> tuple[float, float, float]:
    """R's mean/min/max with na.rm=TRUE; all-NA gives NaN (R would give NaN/Inf/-Inf)."""
    if np.all(np.isnan(track_khz)):
        return float("nan"), float("nan"), float("nan")
    return float(np.nanmean(track_khz)), float(np.nanmin(track_khz)), float(np.nanmax(track_khz))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_specan_features(
    samples: np.ndarray, sample_rate: int, config: SpecanConfig = DEFAULT_CONFIG
) -> dict[str, float]:
    """Compute all 22 specan-style measurements for a mono signal.

    Frequencies are in kHz. ``duration`` is in seconds. ``peakf`` is the frequency
    of the global spectrum maximum (the original voice.csv script hard-coded 0).
    Signals at a rate other than ``config.sample_rate`` are resampled first.
    """
    y = _as_mono_float(samples)
    if y.size == 0:
        raise FeatureExtractionError("Empty signal")
    if not np.all(np.isfinite(y)):
        raise FeatureExtractionError("Signal contains NaN or infinite samples")
    y = _resample_if_needed(y, int(sample_rate), config.sample_rate)
    sr = config.sample_rate
    if len(y) < 2 * config.window_length:
        raise FeatureExtractionError(
            f"Signal too short: {len(y) / sr:.3f} s, need at least {2 * config.window_length / sr:.3f} s"
        )

    freqs_hz, mag = _seewave_spec(y, sr)
    prop = _seewave_specprop(freqs_hz, mag, config.spectral_band_hz)

    fund = _seewave_fund(y, sr, config)
    meanfun, minfun, maxfun = _nan_stats(fund)

    dom = _seewave_dfreq(y, sr, config)
    meandom, mindom, maxdom = _nan_stats(dom)
    dfrange = maxdom - mindom
    modindx = _modulation_index(dom, mindom, maxdom, dfrange)

    return {
        "duration": len(y) / sr,
        "meanfreq": prop["mean"] / 1000.0,
        "sd": prop["sd"] / 1000.0,
        "median": prop["median"] / 1000.0,
        "Q25": prop["Q25"] / 1000.0,
        "Q75": prop["Q75"] / 1000.0,
        "IQR": prop["IQR"] / 1000.0,
        "skew": prop["skewness"],
        "kurt": prop["kurtosis"],
        "sp.ent": prop["sh"],
        "sfm": prop["sfm"],
        "mode": prop["mode"] / 1000.0,
        "centroid": prop["cent"] / 1000.0,
        "peakf": float(freqs_hz[int(np.argmax(mag))]) / 1000.0,
        "meanfun": meanfun,
        "minfun": minfun,
        "maxfun": maxfun,
        "meandom": meandom,
        "mindom": mindom,
        "maxdom": maxdom,
        "dfrange": dfrange,
        "modindx": modindx,
    }


def extract_model_features(
    samples: np.ndarray,
    sample_rate: int,
    config: SpecanConfig = DEFAULT_CONFIG,
    strict: bool = True,
) -> dict[str, float]:
    """Return the 20 model features (voice.csv order) for a mono signal.

    With ``strict=True`` a NaN/inf in any model feature raises
    ``FeatureExtractionError`` (e.g. no voiced frames were detected).
    """
    all_features = compute_specan_features(samples, sample_rate, config)
    model = {name: all_features[name] for name in MODEL_FEATURE_NAMES}
    if strict:
        bad = [name for name, value in model.items() if not np.isfinite(value)]
        if bad:
            raise FeatureExtractionError(
                f"Non-finite feature values {bad}; the recording may contain no voiced speech"
            )
    return model


def features_to_vector(features: dict[str, Any]) -> np.ndarray:
    """Order a feature dict into the model input array of shape (1, 20), float32."""
    missing = [name for name in MODEL_FEATURE_NAMES if name not in features]
    if missing:
        raise FeatureExtractionError(f"Missing features: {missing}")
    vector = np.asarray([features[name] for name in MODEL_FEATURE_NAMES], dtype=np.float32).reshape(1, -1)
    if vector.shape != (1, N_MODEL_FEATURES):
        raise FeatureExtractionError(f"Unexpected feature vector shape {vector.shape}")
    return vector
