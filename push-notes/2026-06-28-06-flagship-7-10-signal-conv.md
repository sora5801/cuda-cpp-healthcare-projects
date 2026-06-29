# Push 2026-06-28 #06 -- flagship 7.10 signal-conv

> Push-note (CLAUDE.md §7.1). Sixth Phase 1 flagship — medical AI.

## 1. Summary

The medical-AI flagship is done: **7.10 Physiological Signal & Waveform Analysis**, a complete, verified GPU
1-D convolution that low-pass-filters a noisy synthetic ECG. It introduces a sixth distinct GPU pattern —
**shared-memory tiling** — the optimization that turns the redundant per-output window reads of a 1-D
convolution into a single tiled load. This 1-D conv is both classical FIR filtering and the core op of every
waveform CNN.

## 2. What changed

- [`projects/07-medical-ai/7.10-physiological-signal-waveform-analysis/`](../projects/07-medical-ai/7.10-physiological-signal-waveform-analysis) — fully implemented:
  - `src/kernels.cu` — `conv1d_kernel` (shared-memory tile + halo, filter in constant memory) + wrapper.
  - `src/reference_cpu.cpp` / `.h` — Gaussian FIR builder + serial convolution.
  - `src/main.cu` — load → build filter → CPU + GPU convolve → verify → print filtered samples.
  - `THEORY.md`, `README.md`, `data/` (synthetic noisy ECG), `scripts/`, `demo/`.
- `docs/STATUS.md` — `7.10` → **done** (6/301). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb)

**7.10 1-D convolution** teaches **shared-memory tiling**: each output sample needs K neighbouring inputs and
adjacent outputs overlap, so a block stages `BLOCK + (K−1)` inputs (a halo on each side) into on-chip shared
memory once, then every thread reads its window from there — eliminating the naive K× global re-reads. The
filter sits in constant memory (warp broadcast), and `__syncthreads()` guards the tile. The standout file is
`src/kernels.cu` (and THEORY §4): the halo-loading scheme and why the barrier is required.

## 4. How to build & run

```powershell
cd projects/07-medical-ai/7.10-physiological-signal-waveform-analysis
msbuild build/physiological-signal-waveform-analysis.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> filtered samples + RESULT: PASS (GPU matches CPU)
```

## 5. What to study here

Reading path: `THEORY.md` (§2 FIR convolution, §4 tiling + halo + `__syncthreads`) → `src/kernels.cu` →
`src/reference_cpu.cpp`. Then try README **Exercises**: write the naive kernel and compare, swap in an ECG
band-pass that preserves QRS, or batch 12-lead records as a 2-D launch.

## 6. Verification

- ✅ `Release|x64` **and** `Debug|x64` build with **zero errors / zero warnings**.
- ✅ Demo **PASS**: deterministic filtered samples match `expected_output.txt`.
- ✅ **GPU == CPU** (`max_abs_err = 5.96e-08`, tol `1e-4`).
- ✅ Low-pass works: RMS of `x − filtered` = 0.091 (noise attenuated).
- ✅ `verify_project.py` → **DONE** (comment ratio **0.66**, no TODOs).
- **Environment:** RTX 2080 (`sm_75`), CUDA 13.3, VS 2026 (`v145`). 2048-sample signal: CPU ~0.06 ms vs GPU
  ~0.55 ms (overhead-bound at this size; the GPU's edge grows with length and with batching many records).

## 7. Known limitations / TODOs

- Generic **Gaussian low-pass** (smooths the narrow R peak); a real ECG pipeline uses a band-pass that
  preserves QRS. Single signal, single precision, zero-padded boundaries.
- One launch over a short signal ⇒ overhead-bound; batching thousands of multi-hour records is the real win.

## 8. Next push preview

Next flagship: **8.03 EEG/MEG spectral processing (cuFFT)** (neuroscience) — a seventh pattern: using a
CUDA **library** (cuFFT) for batched FFTs to compute band-power spectra, with the library call fully explained.
