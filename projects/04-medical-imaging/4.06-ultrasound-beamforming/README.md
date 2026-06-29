# 4.6 — Ultrasound Beamforming

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟢 Beginner · Established** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.6`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

<!-- =======================================================================
     SCAFFOLD STATUS: this README was stamped from the catalog. The prose
     fields below (Deep dive / Algorithms / Datasets / Prior art) are filled
     in from the catalog. Sections marked TODO(impl)/TODO(theory) must be
     completed by the project author before this project is "done"
     (see CLAUDE.md §4.1 and tools/verify_project.py).
     ======================================================================= -->

## Summary

TODO(impl): One paragraph, plain language — what this project does and why a
learner should care. (Seed from the deep dive below.)

## What this computes & why the GPU helps

Delay-and-sum (DAS) beamforming reconstructs B-mode images by computing time-delayed sums of per-element receive signals for every pixel in the image grid. For a 128-element linear array, a 512×512 image, and 4,000 scan lines per second, DAS requires ~3.4 × 10¹⁰ multiply-accumulate operations per second — far beyond real-time CPU capability. GPU parallelism maps each output pixel to a CUDA thread, computes focal delays from element geometry, interpolates raw RF data, and sums across elements; a single RTX-class GPU achieves interactive frame rates for 3D volumetric beamforming. Coherence-based techniques (DMAS, CF) add per-pixel statistics but remain embarrassingly parallel.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Delay-and-sum (DAS), f-k migration, synthetic aperture focusing (SAFT), coherence factor (CF), DMAS (delay-multiply-and-sum), compressed sensing beamforming, Fourier domain reconstruction, adaptive minimum variance beamforming.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/ultrasound-beamforming.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/ultrasound-beamforming.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\ultrasound-beamforming.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, prints the result, shows the
GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/` — a tiny, offline input so the demo runs
  with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` (documented, idempotent).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: Plane-Wave Imaging Challenge in Medical Ultrasound (PICMUS, https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/) — RF data for beamforming evaluation; UltraSound SegLab dataset; IQ ultrasound datasets from open research groups (verify URL at creatis.insa-lyon.fr).

## Expected output

Success looks like `demo/expected_output.txt`. The program computes the result on
both the **GPU** (`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`)
and asserts they agree within the documented tolerance — that agreement is the
correctness guarantee.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports.
2. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
3. [`src/kernels.cu`](src/kernels.cu) — the kernel(s) and host wrapper.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline.
5. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

GPU-accelerated US beamforming repos on GitHub (search "CUDA ultrasound beamforming"); MUST (MATLAB Ultrasound Toolbox, https://www.biomecardio.com/MUST/) — reference DAS + GPU wrappers; Field II (https://field-ii.dk/) — simulation toolbox (CPU, but generates RF data for GPU DAS); k-Wave CUDA (https://github.com/klepo/k-Wave-Fluid-CUDA) — CUDA time-domain acoustic propagation for full-wave ultrasound.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuBLAS for element-weighted summation; custom CUDA kernel: one thread per image pixel, loads element positions into shared memory, vectorized delay computation via `__fmaf_rn`; texture fetch for interpolated RF data; coalesced global memory access across scan-line dimension. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
