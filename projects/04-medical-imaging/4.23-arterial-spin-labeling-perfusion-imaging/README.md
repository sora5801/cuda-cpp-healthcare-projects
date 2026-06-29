# 4.23 — Arterial Spin Labeling & Perfusion Imaging

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.23`
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

Arterial spin labeling (ASL) magnetically labels water protons in arterial blood upstream and images the resulting perfusion-weighted signal difference (labeled minus control). The signal change is only 0.5–1% of background signal, requiring averaging many pairs to achieve adequate SNR; acquisition of dynamic (time-resolved) ASL with 100+ pairs at 2 mm resolution produces datasets where kinetic model fitting (single/multi-delay Buxton model) per voxel is a Bayesian inverse problem amenable to GPU parallelization. Oxford_asl/BASIL uses variational Bayes inference, parallelized across voxels on GPU. 3D multi-delay ASL combined with compressed sensing requires per-timepoint NUFFT reconstruction — same GPU bottleneck as standard CS-MRI.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Buxton kinetic model (single/multi-delay), pulsed ASL (PASL), pseudo-continuous ASL (pCASL), Bayesian kinetic model fitting (BASIL), variational Bayes per voxel, compressed sensing 3D dynamic ASL, T1 partial-volume correction.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/arterial-spin-labeling-perfusion-imaging.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/arterial-spin-labeling-perfusion-imaging.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\arterial-spin-labeling-perfusion-imaging.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: HCP ASL data (https://db.humanconnectome.org/); OpenNeuro ASL datasets (https://openneuro.org/ — search "ASL"); ISMRM 2015 ASL challenge data; UK Biobank ASL pilot data.

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

FSL BASIL (https://fsl.fmrib.ox.ac.uk/fsl/docs/physiological/basil.html) — Bayesian ASL analysis, GPU-parallelizable voxel fits; BART (https://github.com/mrirecon/bart) — dynamic ASL CS reconstruction; ExploreASL (https://github.com/ExploreASL/ExploreASL) — multi-center ASL pipeline; SigPy (https://github.com/mikgroup/sigpy) — dynamic CS-ASL reconstruction.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Per-voxel independent Bayesian fit (one CUDA thread per voxel, Newton-Raphson or variational updates); cuBLAS for kinetic model matrix products; shared memory for model time-course templates; cuFFT for dynamic CS-ASL k-space reconstruction. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
