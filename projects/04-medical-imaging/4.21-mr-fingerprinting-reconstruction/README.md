# 4.21 — MR Fingerprinting Reconstruction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.21`
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

MR Fingerprinting (MRF) acquires a sequence of images with pseudorandom flip angles and TRs; each voxel's signal time course is matched to a dictionary of simulated Bloch-equation evolutions to simultaneously estimate T1, T2, and other parameters. The dictionary (10⁵–10⁶ entries × 1,000 time points) must be searched for each of ~10⁵ voxels, resulting in ~10¹¹ inner products — efficiently computed as a single large matrix-matrix product on GPU (cuBLAS GEMM). Compressed MRF combines partial k-space acquisition with low-rank tensor reconstruction, reducing the GPU workload to manageable batches. Non-Cartesian MRF trajectories require NUFFT-based reconstruction, adding a cuFFT step.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Bloch-simulation dictionary generation, dot-product template matching (inner product per voxel per dictionary entry as GEMM), low-rank subspace reconstruction, ADMM+MRF, physics-driven neural network MRF (DeepMRF), sequence optimization via Cramér-Rao bound.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/mr-fingerprinting-reconstruction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/mr-fingerprinting-reconstruction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\mr-fingerprinting-reconstruction.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: fastMRI MRF (verify URL at fastmri.org); Cleveland Clinic MRF dataset (via IEEE DataPort, verify URL); synthetic MRF datasets generated from XCAT/BrainWeb phantoms; public multi-parametric MRI from qMRI.org (verify URL).

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

BART (https://github.com/mrirecon/bart) — low-rank subspace MRF reconstruction; MRzero (https://github.com/MRsimulator/MRzero) — differentiable MR sequence simulation for MRF design; PyTorch MRF dictionary matching (search GitHub for "MRF dictionary matching PyTorch"); SigPy (https://github.com/mikgroup/sigpy) — NUFFT-based MRF reconstruction.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuBLAS SGEMM for dictionary matching (entire voxel×time matrix vs. dictionary×time matrix); cuFFT for NUFFT in non-Cartesian MRF; GPU-pinned memory for dictionary transfer; batched GEMM across slices via cuBLAS-Xt. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
