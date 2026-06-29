# 7.10 — Physiological Signal & Waveform Analysis

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20AI%20%26%20Clinical%20Deep%20Learning-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 7: Medical AI & Clinical Deep Learning · Catalog ID `7.10`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Filter a physiological waveform (ECG/EEG-like) with a **1-D convolution** — the
single most important operation in waveform analysis. It is both classical signal
filtering (denoising, band-pass) *and* the conv layer at the core of every 1-D
waveform CNN (ResNet/TCN/WaveNet). The GPU lesson here is **shared-memory
tiling**: stage a block of the signal in fast on-chip memory once, then reuse it
across the overlapping output windows. Sixth distinct GPU pattern in the flagships.

## What this computes & why the GPU helps

Continuous high-frequency waveforms (ECG 500–2000 Hz, EEG 256–2048 Hz, ABP, PPG)
are processed for denoising, feature extraction, and classification. The workhorse
is the **1-D temporal convolution** over thousands of time steps. Each output
sample is independent, but adjacent outputs share most of their inputs — so the
naive kernel re-reads each input ~K times from global memory. Tiling into shared
memory removes that redundancy. Clinical pipelines filter **thousands of 24-hour
recordings** in parallel — squarely GPU-bound.

**The parallel bottleneck** is the convolution itself; we use one thread per
output sample with a shared-memory tile of the input (+halo) and the filter in
constant memory.

## The algorithm in brief

- **FIR filter** `h` (here a 31-tap Gaussian low-pass, unity DC gain).
- **Convolution:** `y[n] = Σ_k h[k]·x[n − HALO + k]`, zero-padded at the ends.
- **Tiling:** each block loads `BLOCK + (K−1)` inputs into shared memory, then
  every thread reads its `K`-wide window from there.

See [THEORY.md](THEORY.md) for the signal processing and the tiling analysis.

## Build

Requires **Visual Studio 2026** (v145) + **CUDA 13.3** ([docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/physiological-signal-waveform-analysis.sln`.
2. **`Release|x64`** → **Build** → `build/x64/Release/physiological-signal-waveform-analysis.exe`.

CLI: `msbuild build\physiological-signal-waveform-analysis.sln /p:Configuration=Release /p:Platform=x64`

## Run the demo

```powershell
./demo/run_demo.ps1
```

Filters the committed waveform on CPU + GPU and verifies they match.

## Data

- **Sample (committed):** `data/sample/ecg_sample.txt` — a noisy **synthetic** ECG (2048 samples).
- **Real waveforms:** PhysioNet / MIMIC-IV Waveform / MNE — see
  `scripts/download_data.ps1` and [data/README.md](data/README.md).
- Longer synthetic: `python scripts/make_synthetic.py --n 8192`.

## Expected output

`demo/expected_output.txt` holds the deterministic filtered samples. The GPU
tiled kernel (`src/kernels.cu`) and CPU reference (`src/reference_cpu.cpp`)
convolve in the same order, so they agree to ~`1e-7` (well within the `1e-4`
tolerance).

## Code tour

1. [`src/main.cu`](src/main.cu) — load, build filter, CPU + GPU convolve, verify, print.
2. [`src/kernels.cuh`](src/kernels.cuh) — the tiled-kernel interface + the tiling idea.
3. [`src/kernels.cu`](src/kernels.cu) — the shared-memory tiled kernel (the star) + host wrapper.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the Gaussian filter + serial convolution.

## Prior art & further reading

- **MNE-Python** (<https://github.com/mne-tools/mne-python>) — EEG/MEG processing & filtering.
- **ECG-FM** (<https://github.com/bowang-lab/ecg-fm>) — wav2vec-style ECG foundation model (1-D convs at scale).
- **PhysioNet / WFDB** (<https://github.com/MIT-LCP/wfdb-python>) — waveform I/O and toolkits.
- **cuDNN / PyTorch `conv1d`** — the production 1-D convolution primitives this mirrors.

Study these for the production approach; reimplement didactically (CLAUDE.md §2).

## CUDA pattern used here

**Shared-memory tiling** (block loads input tile + halo once) · filter in
**constant memory** (warp broadcast) · one thread per output sample · dynamic
shared memory sized to the tile. This is exactly the inner loop of a 1-D conv
layer.

## Exercises

1. **Naive vs tiled.** Write the naive kernel (read K inputs from global memory
   per output) and compare its time to the tiled one as `K` grows.
2. **Band-pass for ECG.** Replace the Gaussian low-pass with a band-pass FIR
   (~0.5–40 Hz) that preserves the QRS complex; design it with a windowed sinc.
3. **Batched multi-lead.** Filter 12 leads × many records at once (a 2-D launch),
   the real clinical throughput case.
4. **Separable 2-D.** Extend to a 2-D separable convolution (filter rows then
   columns) — the bridge to image filtering.
5. **`__restrict__` & unroll.** Measure the effect of `#pragma unroll` on the tap
   loop and of marking pointers `__restrict__`.

## Limitations & honesty

- The filter is a **generic Gaussian low-pass**; a real ECG pipeline uses a
  band-pass that preserves QRS (this one smooths the narrow R peak — see the demo).
- Single signal, single precision; production batches thousands of records.
- No learned filters here — but this *is* the 1-D conv that a waveform CNN learns.
