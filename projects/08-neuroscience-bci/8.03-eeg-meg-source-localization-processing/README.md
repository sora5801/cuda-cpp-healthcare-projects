# 8.03 — EEG/MEG Spectral Processing (cuFFT)

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Neuroscience%20%26%20Brain--Computer%20Interfaces-lightgrey)

> **🟢 Beginner · Established** — Domain 8: Neuroscience & Brain-Computer Interfaces · Catalog ID `8.03`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Compute the **frequency content** of multi-channel EEG and its clinical **band
powers** (delta/theta/alpha/beta/gamma) using the GPU FFT library **cuFFT**. cuFFT
batches one real-to-complex FFT per channel in a single call; a tiny custom kernel
takes the magnitude-squared (the power spectrum). This is the seventh distinct GPU
pattern in the flagships: **using a CUDA library** — and doing it without the
library being a black box.

## What this computes & why the GPU helps

Quantitative EEG/MEG analysis transforms each channel to the frequency domain and
integrates power into bands. Real montages have 64–306 channels at 1–10 kHz over
long recordings; the FFTs are independent across channels and analysis windows,
which **cuFFT** batches efficiently. The naive DFT is `O(N²)`; the FFT is
`O(N log N)` — so the GPU win here is both **parallelism** and a better
**algorithm** (the demo shows ~16× even on a tiny 8×256 case).

**The parallelized work** is the batched FFT (cuFFT) plus a per-bin power kernel;
band integration is a cheap host post-step.

## The algorithm in brief

- **FFT** each channel (real → complex): `X_c[k] = Σ_t x_c[t] e^{-2πi k t/N}`.
- **Power:** `P_c[k] = |X_c[k]|² / N²`.
- **Bands:** sum `P_c[k]` over the bins whose frequency `k·fs/N` falls in each band.

See [THEORY.md](THEORY.md) for the spectral math, the cuFFT R2C layout, and the inverse problem (source localization).

## Build

Requires **Visual Studio 2026** (v145) + **CUDA 13.3** ([docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).
This project **links cuFFT** (`cufft.lib`, already in the `.vcxproj`).

1. Open `build/eeg-meg-source-localization-processing.sln`.
2. **`Release|x64`** → **Build** → `build/x64/Release/eeg-meg-source-localization-processing.exe`.

CLI: `msbuild build\eeg-meg-source-localization-processing.sln /p:Configuration=Release /p:Platform=x64`

## Run the demo

```powershell
./demo/run_demo.ps1
```

Computes band powers on CPU + GPU and verifies they match.

## Data

- **Sample (committed):** `data/sample/eeg_sample.txt` — 8-channel **synthetic** EEG.
- **Real EEG/MEG:** PhysioNet / MNE / OpenNeuro — see `scripts/download_data.ps1`
  and [data/README.md](data/README.md).
- Longer window: `python scripts/make_synthetic.py --n 512 --fs 512`.

## Expected output

`demo/expected_output.txt` holds the deterministic per-channel band powers and
dominant band. cuFFT (`src/kernels.cu`) and the naive DFT (`src/reference_cpu.cpp`)
agree to ~`1e-6` relative; the dominant band of each synthetic channel is recovered
exactly.

## Code tour

1. [`src/main.cu`](src/main.cu) — load, CPU DFT + GPU cuFFT, band powers, verify, print.
2. [`src/kernels.cuh`](src/kernels.cuh) — the cuFFT interface + the "library, not black box" note.
3. [`src/kernels.cu`](src/kernels.cu) — **the cuFFT call (fully documented)** + the power kernel.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the naive DFT + band integration.

## Prior art & further reading

- **MNE-Python** (<https://github.com/mne-tools/mne-python>) — comprehensive EEG/MEG analysis.
- **FieldTrip** (<https://github.com/fieldtrip/fieldtrip>) — MATLAB MEG/EEG toolbox.
- **EEGLAB** (<https://github.com/sccn/eeglab>) — EEG analysis ecosystem.
- **cuFFT** (<https://developer.nvidia.com/cufft>) — the library this project uses.

Study these for the production approach; reimplement didactically (CLAUDE.md §2).

## CUDA pattern used here

**Using a CUDA library (cuFFT)** for batched real-to-complex FFTs · a small custom
power kernel for `|X|²` · the library call explained, not hidden · CPU verified
against a naive DFT. The same `cufftPlan1d(…, CUFFT_R2C, batch)` underlies any
spectral pipeline.

## Exercises

1. **Welch's method.** Split a long recording into overlapping windows, apply a
   Hann taper, FFT each (a bigger cuFFT batch), and average the power spectra.
2. **Spectrogram.** Compute a short-time Fourier transform (STFT) and plot
   time × frequency — the basis of seizure/sleep staging.
3. **cuFFT plan reuse.** Create the plan once and reuse it across many windows;
   measure the saving versus re-planning each call.
4. **Inverse FFT filter.** Zero out-of-band bins and `cufftExecC2R` back to time —
   a frequency-domain band-pass; compare to the time-domain FIR of `7.10`.
5. **Source localization.** Implement a simple LCMV beamformer over the band-limited
   data (the inverse problem from the catalog deep-dive).

## Limitations & honesty

- **Spectral processing only.** The catalog entry also covers **source localization**
  (the EEG/MEG inverse problem); this flagship focuses on the cuFFT spectral pipeline
  and describes localization in THEORY.
- Synthetic clean sinusoids; no tapering/windowing (a real pipeline uses Hann/Welch),
  single window, single precision (cuFFT default).
