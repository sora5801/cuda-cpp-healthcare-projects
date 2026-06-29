# Push 2026-06-29 #00 -- flagship 8.03 cufft-eeg

> Push-note (CLAUDE.md §7.1). Seventh Phase 1 flagship — neuroscience / BCI. First flagship to link a CUDA library.

## 1. Summary

The neuroscience flagship is done: **8.03 EEG/MEG Spectral Processing (cuFFT)**, a complete, verified GPU
spectral-analysis pipeline that computes per-channel EEG band powers (delta/theta/alpha/beta/gamma). It
introduces a seventh distinct GPU pattern — **using a CUDA library (cuFFT)** — and does so per the "no black
boxes" rule: the `cufftExecR2C` call is fully documented (what it computes, its batched layout, what
hand-rolling would take). It is also the first flagship to link an external CUDA library (`cufft.lib`).

## 2. What changed

- [`projects/08-neuroscience-bci/8.03-eeg-meg-source-localization-processing/`](../projects/08-neuroscience-bci/8.03-eeg-meg-source-localization-processing) — fully implemented:
  - `src/kernels.cu` — batched **cuFFT** R2C (one FFT per channel) + a `power_kernel` for `|X|²`.
  - `src/reference_cpu.cpp` / `.h` — naive `O(N²)` DFT reference + EEG band integration.
  - `src/main.cu` — load → CPU DFT + GPU cuFFT → band powers → verify → print per-channel bands + dominant.
  - `build/*.vcxproj` now links **`cufft.lib`** (both configs); `CMakeLists.txt` links `CUDA::cufft`.
  - `THEORY.md`, `README.md`, `data/` (synthetic multi-band EEG), `scripts/`, `demo/`.
- `docs/STATUS.md` — `8.03` → **done** (7/301). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb)

**8.03 cuFFT EEG** teaches **how to use a CUDA library well**: `cufftPlan1d(.., CUFFT_R2C, n_ch)` plans
`n_ch` independent real FFTs and `cufftExecR2C` runs them all in one `O(N log N)` batched call — versus the
reference's transparent `O(N²)` DFT. The standout file is `src/kernels.cu` (and THEORY §4): the exact R2C
layout (`cufftComplex == float2`, `N/2+1` bins via Hermitian symmetry), the normalization, and what the
library encapsulates — documented, not cargo-culted.

## 4. How to build & run

```powershell
cd projects/08-neuroscience-bci/8.03-eeg-meg-source-localization-processing
msbuild build/eeg-meg-source-localization-processing.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> per-channel band powers + dominant band + RESULT: PASS
```

## 5. What to study here

Reading path: `THEORY.md` (§2 DFT/FFT + bands, §4 the cuFFT call) → `src/kernels.cu` →
`src/reference_cpu.cpp`. Then try README **Exercises**: Welch averaging, an STFT spectrogram, plan reuse, an
inverse-FFT band-pass, or an LCMV beamformer (the source-localization inverse problem).

## 6. Verification

- ✅ `Release|x64` **and** `Debug|x64` build with **zero errors / zero warnings** (cuFFT links in both).
- ✅ Demo **PASS**: deterministic per-channel band powers match `expected_output.txt`.
- ✅ **cuFFT vs naive DFT** agree to `1.8e-06` relative (allclose atol=1e-6, rtol=1e-3).
- ✅ Every synthetic channel's **dominant band recovered exactly** (ch0→alpha, ch1→beta, ch2→theta,
  ch3→delta, ch4→gamma, ...).
- ✅ `verify_project.py` → **DONE** (comment ratio **0.63**, no TODOs).
- **GPU + algorithmic win:** CPU naive DFT ~3.39 ms vs GPU cuFFT+power ~0.21 ms (~16×); the gap explodes
  with N (`O(N²)` → `O(N log N)`) and channel count.
- **Environment:** RTX 2080 (`sm_75`), CUDA 13.3, VS 2026 (`v145`).

## 7. Known limitations / TODOs

- **Spectral processing only**: the catalog entry also spans **source localization** (the inverse problem);
  this flagship focuses on the cuFFT spectral pipeline and describes localization in THEORY.
- Synthetic clean sinusoids; no windowing/tapering (real pipelines use Hann/Welch); single window, single
  precision (cuFFT default).

## 8. Next push preview

Next flagship: **9.02 Compartmental / metapopulation ODE ensembles** (epidemiology) — an eighth pattern:
**ensemble ODE integration** (thousands of SEIR parameter sets integrated in parallel, one thread per
trajectory, RK4).
