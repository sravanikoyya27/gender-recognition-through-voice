# Acoustic feature specification

This document is the contract for the feature vector consumed by the gender classifier. The implementation is `ml_models/voice_features/specan.py`; both `ml_models/data_preparation.py` (training data) and `backend/app/ml/features.py` (inference) import it, so training and serving can never drift apart.

## 1. Historical origin

| Source | What it did |
|---|---|
| B.Tech project R script (`R Code.txt`, report chapter 5) | `warbleR::specan(X, bp = c(0, 22))` on each `.wav`, exported to a spreadsheet |
| Training data of the project's Keras MLP | Kaggle/GitHub `voice.csv` (3 168 rows, 20 features + label), created by Kory Becker's `specan3()` R function, itself a modified copy of `warbleR::specan` built on the `seewave` package |
| Reference paper 2 (Buyukyilmaz & Cibikdiken, 2016) | Same `voice.csv`; Table II lists the 22 specan measurements, of which 20 were used |

`warbleR::specan` returns 22 measurements. `voice.csv` keeps 20 of them: `duration` was dropped because every clip was cut to the same length, and `peakf` was dropped because the script that produced the dataset hard-coded it to 0 (the `fpeaks` call is commented out in `specan3`). The Phase 1 backend placeholder already fixed those 20 names in `voice.csv` column order, and this phase keeps that order as the model contract.

## 2. Final model feature vector (20 values, this exact order)

All frequencies are in **kHz**, matching `voice.csv`. `sp.ent`, `sfm`, `skew`, `kurt` and `modindx` are dimensionless.

| # | Name | Unit | Definition (seewave function) | Notes |
|---|---|---|---|---|
| 1 | `meanfreq` | kHz | `specprop$mean`: Σ p(f)·f over the 0–280 Hz band, where p is the normalised magnitude spectrum | |
| 2 | `sd` | kHz | `specprop$sd`: √Σ p(f)·(f − mean)² | |
| 3 | `median` | kHz | `specprop$median`: first bin where cumulative p exceeds 0.5 (R index convention, see §4) | |
| 4 | `Q25` | kHz | `specprop$Q25`: first bin where cumulative p exceeds 0.25 | |
| 5 | `Q75` | kHz | `specprop$Q75`: first bin where cumulative p exceeds 0.75 | |
| 6 | `IQR` | kHz | `Q75 − Q25` | |
| 7 | `skew` | – | `specprop$skewness`: skewness of the **amplitude values** of the band, `(Σz³/(L−1)) / s³` with `z = p − mean(p)`, `s = sd(p)` | Historical quirk: a moment of the bin amplitudes, not of the frequency distribution |
| 8 | `kurt` | – | `specprop$kurtosis`: `(Σz⁴/(L−1)) / s⁴` | Same quirk; not excess kurtosis |
| 9 | `sp.ent` | – | `specprop$sh` = seewave `sh()`: Shannon entropy of p, normalised by log(L), zeros replaced by 1e‑7 | Range (0, 1] |
| 10 | `sfm` | – | `specprop$sfm` = seewave `sfm()`: geometric mean / arithmetic mean of p, zeros replaced by 1e‑5 | Range (0, 1] |
| 11 | `mode` | kHz | `specprop$mode`: frequency of the largest bin in the band | |
| 12 | `centroid` | kHz | `specprop$cent`: Σ f·p(f) | Identical formula to `meanfreq`; the two columns are equal in `voice.csv` and in this implementation |
| 13 | `meanfun` | kHz | mean of the `seewave::fund` track (NA frames ignored) | Cepstral fundamental, see §3.3 |
| 14 | `minfun` | kHz | min of the `fund` track | |
| 15 | `maxfun` | kHz | max of the `fund` track | |
| 16 | `meandom` | kHz | mean of the `seewave::dfreq` track (NA frames ignored) | Dominant frequency, see §3.4 |
| 17 | `mindom` | kHz | min of the `dfreq` track | |
| 18 | `maxdom` | kHz | max of the `dfreq` track | |
| 19 | `dfrange` | kHz | `maxdom − mindom` | |
| 20 | `modindx` | – | mean of \|y[j] − y[j+1]\| over non‑NA frames j of the `dfreq` track, divided by `dfrange`; 0 when `mindom == maxdom` | |

Two further measurements are computed and stored in `voice_features_full.csv` as metadata but are **not** model inputs:

| Name | Unit | Definition |
|---|---|---|
| `duration` | s | `len(signal) / sample_rate` |
| `peakf` | kHz | frequency of the maximum of the whole-signal spectrum (the historical script wrote 0) |

## 3. Implementation, step by step

Parameters (`SpecanConfig` defaults) reproduce the `specan3` call used for `voice.csv`: `wl = 2048`, `threshold = 5`, `fmax = 280`, `flim = c(0, 0.28)`, `bp = c(0, 22)`, `ovlp = 50` for `fund` and `0` for `dfreq`.

### 3.1 Input normalisation

* Any input is converted to mono float64; multi-channel input is averaged.
* Signals not at **16 000 Hz** are resampled with `librosa.resample`. All analysis runs at 16 kHz.
* Minimum length: 2 × `wl` = 4 096 samples (0.256 s). Data preparation additionally enforces 1–10 s per file.

### 3.2 Whole-signal spectrum and band statistics (`seewave::spec` + `specprop`)

1. Multiply the whole signal by a symmetric Hann window (`ftwindow(n, "hanning")`).
2. One FFT of length `n = len(signal)`; keep the magnitude of bins `0 … n/2 − 1` (Nyquist dropped, as `spec` does); normalise so the maximum is 1. Implemented as a single-frame `librosa.stft(n_fft = n, hop = n, center = False)`.
3. Select bins with `0 ≤ f ≤ 280 Hz`, divide by their sum to get a probability mass function `p`.
4. Compute rows 1–12 of the table with the `specprop` formulas above; convert Hz to kHz.

### 3.3 Fundamental frequency track (`seewave::fund`)

1. `afilter`: set every sample with `|x| ≤ 5 % · max|x|` to 0, then replace zeros by 1e‑6.
2. Frames of `wl = 2048` samples with hop `1024` (50 % overlap), no window, starting positions `0, 1024, …` exactly as R's `seq(1, n − wl, by = hop)` (see §4).
3. Per frame: cepstrum `c = Re(IFFT(log |FFT(frame)|))`, keep quefrencies `0 … wl/2 − 1`, non-finite values → 0.
4. Search the peak at quefrencies `q ≥ fmaxi = floor(16000 / 280) = 57`. If the peak is at the first allowed quefrency the frame has no fundamental (NA). Otherwise `f0 = 16000 / q` Hz.
5. `meanfun`, `minfun`, `maxfun` are the mean/min/max over frames with a value, in kHz.

### 3.4 Dominant frequency track (`seewave::dfreq`)

1. `afilter` as above (no 1e‑6 replacement).
2. STFT with `wl = 2048`, hop `2048` (0 % overlap), symmetric Hann window, frames as in §3.3; keep the first `wl/2` bins.
3. Bandpass: specan clips `bp[2]` to `ceiling(sr/2000) − 1` kHz, so at 16 kHz the band is **0–7 kHz**; bins outside are zeroed.
4. Per frame the arg‑max bin gives the dominant frequency `(bin index) · 16000 / 2048` Hz. A maximum at bin 0 (silent frame or DC) is NA.
5. `meandom`, `mindom`, `maxdom`, `dfrange`, `modindx` follow from the track as in the table.

## 4. Index and framing conventions reproduced from R

* **Quantiles**: R uses `freq[length(cumamp[cumamp <= q]) + 1]` (1‑based). In 0‑based terms that is `freq[count(cum <= q)]`; the tests pin this with a 4-bin toy spectrum.
* **Frame count**: R's `seq(1, n − wl, by = hop)` never starts a frame at position `n − wl + 1`. The Python code frames `signal[: n − 1]`, which yields the identical count and start positions (tested against the R formula for several lengths).
* **NA handling**: frame values that R would set to NA are NaN and excluded with `nanmean`/`nanmin`/`nanmax`. If *every* frame is NA the feature is NaN; `extract_model_features(strict=True)` then raises `FeatureExtractionError` instead of returning garbage, and `data_preparation` skips the file with reason `non_finite`.

## 5. Known differences from the historical output

| Aspect | Historical (`voice.csv` / warbleR) | This implementation | Effect |
|---|---|---|---|
| Sample rate | Mixed sources; some files at 44.1 kHz (`maxdom` up to 21.9 kHz in `voice.csv`) | Fixed 16 kHz, so the dominant band is 0–7 kHz | `meandom`/`maxdom` for a 44.1 kHz source differ from a historical row of the same recording; within our own pipeline training and inference are consistent |
| Clip length | 20 s segments (`start = 0, end = 20`) | 1–10 s clips | Whole-signal FFT resolution is coarser (0.1–1 Hz instead of 0.05 Hz); band statistics are robust to this, `skew`/`kurt`/`sfm` shift somewhat |
| `peakf` | Hard-coded 0 | Real spectral peak | Not a model input, so no effect on the model |
| Hann window | Symmetric (`ftwindow`) | Symmetric (`scipy.signal.windows.hann(sym=True)`) | None |
| FFT library | R `fft` / FFTW | numpy / scipy via librosa | Floating-point noise only |
| Numeric parity | – | **Not verified against R output.** R is not installed and the original audio is unavailable, so parity is established at the formula level and by the R-convention unit tests, not by comparing numbers on the same file | Blocker noted in the phase report |

## 6. Scaling

Raw feature values are what the CSVs contain. `save_splits()` fits a `StandardScaler` on the **training split only** and saves it as `scaler.joblib`; `*_scaled.csv` copies and `DataPreparation.extract_features(..., scaler=...)` apply it. The same scaler artefact must be loaded by the inference service (`backend/app/config.py: SCALER_PATH`).
