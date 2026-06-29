# 5.5 — Deformable Dose Accumulation & Adaptive Radiotherapy

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Radiation%20Therapy%20%26%20Medical%20Physics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 5: Radiation Therapy & Medical Physics · Catalog ID `5.5`
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

Adaptive radiotherapy (ART) adjusts the treatment plan during a course of fractions based on daily imaging (CBCT), requiring: (1) daily GPU CBCT reconstruction, (2) deformable image registration (DIR) between planning CT and daily image, (3) deformable warping of the dose distribution via the DVF to accumulate physically meaningful total dose. DIR and dose warping on a 512³ volume require iterative GPU Demons/B-spline followed by trilinear interpolation of the 3D DVF — each voxel's dose is mapped to its deformed position. Online ART workflows (MR-Linac) must complete all steps in <5 min, achievable only with GPU. Uncertainty in DIR propagates to dose uncertainty, motivating ensemble DIR and probabilistic dose accumulation on GPU.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Diffeomorphic Demons DIR, B-spline FFD, VoxelMorph for daily DIR, trilinear DVF warp for dose accumulation, summation-of-deformed-doses vs. energy-mass-transfer method, DIR uncertainty quantification, plan re-optimization on adapted anatomy.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/deformable-dose-accumulation-adaptive-radiotherapy.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/deformable-dose-accumulation-adaptive-radiotherapy.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\deformable-dose-accumulation-adaptive-radiotherapy.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: TCIA CT-on-rails / CBCT datasets; DIR-Lab 4D-CT lung dataset (https://www.dir-lab.com/); AAPM TG-132 DIR test cases; CREATIS deformable lung phantom (https://www.creatis.insa-lyon.fr/).

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

Plastimatch (https://plastimatch.org/) — GPU B-spline DIR + dose warping, DICOM-RT; VoxelMorph (https://github.com/voxelmorph/voxelmorph) — DL DIR for daily CBCT to CT; CERR (https://github.com/cerr/CERR) — deformable dose accumulation pipeline; pyRadPlan (https://github.com/e0404/pyRadPlan) — adaptive plan re-optimization.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

GPU Demons iterative kernel (per-voxel force computation + Gaussian smoothing via cuFFT); custom CUDA trilinear warp for dose mapping; cuBLAS for B-spline coefficient computation; CUDA atomic adds for accumulated dose histogram. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
