# 4.10 — Super-Resolution Microscopy Reconstruction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.10`
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

STORM/PALM single-molecule localization microscopy (SMLM) acquires thousands of diffraction-limited frames; each fluorophore's sub-pixel position is estimated by fitting a 2D Gaussian PSF to sparse activated emitters. Processing 10⁴–10⁵ raw frames at 512×512 per acquisition demands massively parallel PSF fitting — each detected spot is independent, creating an embarrassingly parallel workload ideal for GPU. SRRF (Super-Resolution Radial Fluctuations) and SOFI (Second-Order Fluctuation Imaging) compute cross-correlations or cumulants over time stacks, with O(N·T) operations per pixel. Structured Illumination Microscopy (SIM) reconstruction requires per-orientation/phase FFT and OTF inversion, naturally parallelizable across k-space.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Gaussian/PSF maximum-likelihood fitting (SMLM), SRRF (radial fluctuation analysis), SOFI (cumulant imaging), SIM reconstruction (OTF inversion, Wiener filter), deconvolution (Richardson-Lucy + GPU), DECODE (deep probabilistic SMLM localization).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/super-resolution-microscopy-reconstruction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/super-resolution-microscopy-reconstruction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\super-resolution-microscopy-reconstruction.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: EPFL SMLM Challenge dataset (https://srm.epfl.ch/srm/dataset/challenge-2016/) — synthetic and real STORM/PALM frames; BioImage Archive SMLM collections (https://www.ebi.ac.uk/biostudies/bioimages); OpenMicroscopy Environment (OME-TIFF standard).

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

DECODE (https://github.com/TuragaLab/DECODE) — deep learning GPU SMLM localizer, orders of magnitude faster than MLE; ThunderSTORM (FIJI plugin, GPU-optional); NanoJ-SRRF (https://github.com/HenriquesLab/NanoJ-SRRF) — GPU-accelerated SRRF in ImageJ; fairSIM (https://github.com/fairSIM/fairSIM) — GPU-enabled SIM reconstruction.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom CUDA kernels for per-emitter Gaussian fitting (one warp per candidate emitter); cuFFT for SIM phase/OTF; shared memory for 7×7 PSF patch fitting; atomic additions for localization histogram accumulation. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
