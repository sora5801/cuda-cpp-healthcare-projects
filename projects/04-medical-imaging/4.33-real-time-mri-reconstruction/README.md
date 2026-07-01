# 4.33 — Real-Time MRI Reconstruction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.33`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Interventional and cardiac MRI need a new image every ~50–100 ms so a clinician can
watch a beating heart or steer a catheter in real time. Modern scanners do this by
acquiring k-space along **radial spokes at the golden angle** and reconstructing a fresh
frame from a **sliding window** of the most recent spokes. Radial samples don't sit on
the FFT grid, so each frame is a **gridding NUFFT**: density-compensate the samples,
spread them onto a Cartesian grid with a Kaiser-Bessel kernel, inverse-FFT with cuFFT,
and deapodize. This project builds that whole pipeline as a small, heavily-commented
CUDA program, reconstructs a 6-frame movie of a moving synthetic phantom, and proves the
GPU result matches a plain-C++ reference exactly.

## What this computes & why the GPU helps

Interventional and cardiac MRI require image reconstruction latency <100 ms to enable
real-time guidance (catheter navigation, cardiac function monitoring). Online adaptive
reconstruction with a sliding window (or XD-GRASP) processes continuously acquired
non-Cartesian k-space (radial, spiral) with a GPU NUFFT running in a locked pipeline
with acquisition. Frameworks like Gadgetron pipeline coil compression, NUFFT, and DL
inference on the GPU acquisition-synchronously; the cardiac cycle adds a gating
dimension that pushes reconstruction to interactive speeds only feasible on a GPU.

**The parallel bottleneck:** the **gridding scatter**. Every acquired k-space sample
(thousands per window) is spread independently onto the ~`(W+1)²` nearest grid cells —
an embarrassingly parallel scatter we map to **one GPU thread per sample**. The inverse
FFT that follows is a solved problem handled by **cuFFT**. Together they dominate the
per-frame cost; both parallelize cleanly, which is why a GPU can hit the real-time
latency budget that a serial CPU cannot at clinical grid/coil sizes.

## The algorithm in brief

- **Golden-angle radial acquisition** — spokes 111.25° apart so any window tiles k-space.
- **Density compensation** — weight each sample by `|k|` (the radial ramp filter).
- **Kaiser-Bessel gridding** — convolution-interpolate samples onto the Cartesian grid.
- **ifftshift → inverse FFT (cuFFT) → fftshift** — transform the grid to a centered image.
- **Deapodization** — divide by the KB kernel's analytic Fourier transform.
- **Sliding window** — repeat per frame over a window that advances by `stride` spokes.
- **Deterministic scatter** — accumulate the grid in **fixed-point integers** so the GPU
  atomics are order-independent and bit-match the CPU.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)). The project links **cuFFT**
(for the per-frame inverse 2D FFT) — already wired into both `.vcxproj` link sections
and `CMakeLists.txt`.

1. Open `build/real-time-mri-reconstruction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/real-time-mri-reconstruction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\real-time-mri-reconstruction.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (uses the optional CMake build)
```

The demo builds if needed, runs on `data/sample/radial_sample.txt`, prints the result,
shows the GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/radial_sample.txt` — a tiny, offline synthetic
  radial acquisition so the demo runs with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` (documented, idempotent; prints
  registration instructions, never bypasses credentials).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: cardiac MRI from the **ACDC** challenge
(<https://www.creatis.insa-lyon.fr/Challenge/acdc/>) and **CMRxRecon 2023**
(<https://cmrxrecon.github.io/>); real-time cardiovascular raw data via **OCMR**
(<https://ocmr.info/>). All require registration and cannot be redistributed here, so the
committed sample is synthetic and clearly labeled.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The program
reconstructs the movie on both the **GPU** (`src/kernels.cu`) and a **CPU reference**
(`src/reference_cpu.cpp`) and asserts they agree within a documented tolerance — that
agreement is the correctness guarantee. Because the gridding uses fixed-point integer
accumulation, the two paths match to ~`1e-11` (only the cuFFT-vs-radix-2-FFT rounding
differs). A second, stronger check confirms the last frame's image correlates with the
known synthetic phantom (correlation ≈ 0.96), i.e. the reconstruction really recovered
the anatomy — and the per-frame peak location shifts as the phantom moves, proving the
movie is dynamic.

## Code tour

Read in this order:

1. [`src/grid_core.h`](src/grid_core.h) — the shared `__host__ __device__` math:
   `Cplx`, the Kaiser-Bessel weight, its deapodization Fourier transform, the `|k|`
   density-compensation factor, the golden angle, and the fixed-point quantizers.
2. [`src/reference_cpu.h`](src/reference_cpu.h) — the data model (`RadialData`) and the
   reconstruction API; read the header block first for the "what & why".
3. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU per frame, verifies, reports.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the scatter/thread mapping.
5. [`src/kernels.cu`](src/kernels.cu) — the gridding **scatter kernel** (fixed-point
   atomics), the ifftshift/fold kernel, the **cuFFT** call, and the deapodize+magnitude
   kernel.
6. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline: a
   radix-2 FFT and the same gridding loop the kernels run.
7. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, host I/O helpers.

## Prior art & further reading

- **Gadgetron** (<https://github.com/gadgetron/gadgetron>) — the reference open-source
  *streaming* MR reconstruction server; study how it pipelines coil compression, NUFFT,
  and inference acquisition-synchronously.
- **BART** (<https://github.com/mrirecon/bart>) — the Berkeley Advanced Reconstruction
  Toolbox; its `nufft`/`nufftbase` and `pics` tools are the gold standard for gridding
  and GRASP-style reconstruction. Read its Kaiser-Bessel + density-compensation code.
- **SigPy** (<https://github.com/mikgroup/sigpy>) — a readable Python NUFFT/CUDA library;
  great for prototyping and for seeing gridding written clearly.
- **MRzero** (<https://github.com/MRsimulator/MRzero>) — differentiable MR simulation, if
  you want to explore learned reconstruction on top of this pipeline.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Scatter + atomic reduce** (PATTERNS.md §1: "clustering / centroid accumulation") for
the gridding, plus **cuFFT** for the inverse transform. One thread owns one radial
sample and atomically spreads its Kaiser-Bessel-weighted, density-compensated value onto
the grid; the grid is accumulated in **fixed-point integers** (PATTERNS.md §3) so the
atomics are deterministic and bit-match the CPU. Per-pixel fold/ifftshift, scale, and
deapodize+magnitude kernels are simple one-thread-per-pixel maps. In production the
per-frame kernels would overlap with acquisition via **CUDA streams** (double-buffering:
acquire the next spoke while reconstructing the current frame); this teaching version
runs them sequentially and synchronously so each stage is visible.

## Exercises

1. **Grid oversampling.** Grid onto a `2n×2n` grid, inverse-FFT, and crop the central
   `n×n`. Measure how the correlation-with-truth and the streaking change (this is the
   standard way real gridders suppress aliasing).
2. **Better density compensation.** Replace the analytic `|k|` ramp with an *iterative*
   DCF (Pipe–Menon): grid a field of ones, un-grid it, and invert. Compare image quality.
3. **Window size vs. temporal resolution.** Sweep `--win` (fewer spokes → sharper motion
   but more streaks; more spokes → smoother but blurrier). Plot the trade-off.
4. **A real trajectory.** Add a **spiral** trajectory to `make_synthetic.py` and
   `sample_kpos`, and reconstruct it with the *same* gridding kernel (only the geometry
   changes) — the payoff of the gridding abstraction.
5. **Stream the pipeline.** Split the per-frame work across two CUDA streams so frame
   `f+1`'s scatter overlaps frame `f`'s FFT; measure the throughput change with Nsight.

## Limitations & honesty

- **Reduced-scope teaching version** (CLAUDE.md §13). It is a single 2D slice, a single
  receive coil, FP32, a 32×32 grid, and a simple `|k|` density-compensation with **no
  grid oversampling**. Real systems use 3D + cardiac-phase volumes, 8–32 coils with
  parallel-imaging (GRAPPA/SENSE), iterative DCF, 1.5–2× oversampling, and compressed-
  sensing/low-rank priors (GRASP, XD-GRASP, L+S) — described in THEORY "real world".
- **Streaking & blur are real.** With only ~21 spokes and no oversampling the images show
  radial streak artifacts; the correlation-with-truth (~0.96) is good but not perfect,
  and we say so rather than hiding it. The verification tolerances are honest (THEORY
  "verify correctness").
- **The data is synthetic.** The phantom (a few Gaussian blobs, one bobbing) is a
  mathematical stand-in for anatomy + motion, labeled synthetic everywhere. **No clinical
  validity is implied.**
- **Brightness scale is arbitrary.** Gridding + density compensation leave an undetermined
  overall gain, so the demo normalizes the displayed image to peak = 1.0 (standard for MR
  display); the raw MR-unit peak is printed for transparency.
