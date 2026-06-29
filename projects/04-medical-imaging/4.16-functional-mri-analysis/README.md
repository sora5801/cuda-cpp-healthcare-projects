# 4.16 — Functional MRI Analysis

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.16`
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

fMRI BOLD signal analysis involves preprocessing pipelines (motion correction, slice-timing, smoothing, registration) and statistical modeling (general linear model, GLM) across hundreds of thousands of voxels and thousands of time points. ICA (independent component analysis) via MELODIC decomposes a T × V spatiotemporal matrix; for 1,200 TRs and 150,000 gray-matter voxels, the matrix-SVD and subsequent unmixing are natural cuBLAS workloads. Resting-state functional connectivity computes a V × V correlation matrix — for 100,000 voxels this is a 10¹⁰-element matrix — computed efficiently on GPU via batched inner products. Dynamic functional connectivity via sliding-window or HMM approaches further multiply this cost, requiring GPU for tractable runtimes.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

GLM (HRF convolution and OLS/WLS per voxel), ICA (MELODIC), seed-based connectivity, graph-theoretic brain network analysis, HMM dynamic connectivity, diffusion embedding, CNN/transformer resting-state biomarker extraction, k-means parcellation on GPU.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/functional-mri-analysis.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/functional-mri-analysis.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\functional-mri-analysis.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: HCP fMRI (https://db.humanconnectome.org/) — resting-state and task fMRI, 7T/3T; OpenFMRI / OpenNeuro (https://openneuro.org/) — thousands of fMRI datasets in BIDS; ABIDE autism fMRI (http://fcon_1000.projects.nitrc.org/indi/abide/); UK Biobank fMRI (https://www.ukbiobank.ac.uk/).

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

FSL (https://fsl.fmrib.ox.ac.uk/) — MELODIC GPU ICA, FEAT GLM, BEDPOSTX; Nilearn (https://nilearn.github.io/) — Python fMRI statistical learning with scikit-learn; BrainSpace (https://github.com/MICA-MNI/BrainSpace) — gradient analysis on GPU; fMRIPrep (https://github.com/nipreps/fmriprep) — standardized preprocessing pipeline (CUDA-accelerated ANTs registration within).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuBLAS for GLM design-matrix product (V × T × T × T^-1 × T × V batched); cuSOLVER for ICA SVD; cuRAND for permutation testing; GPU histogram for parcellation; multi-GPU via PyTorch for DL resting-state classifiers. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
