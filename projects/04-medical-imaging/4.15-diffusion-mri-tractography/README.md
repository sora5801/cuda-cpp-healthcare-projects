# 4.15 — Diffusion MRI & Tractography

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.15`
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

Diffusion MRI models water diffusion anisotropy in tissue to map white-matter fiber orientations. Fitting diffusion models (DTI, DKI, NODDI) per voxel is trivially parallel — each voxel is independent — and for a 2 mm isotropic brain (~10⁵ voxels × 100 diffusion directions), batch GPU fitting is 50–100× faster than serial CPU. Constrained spherical deconvolution (CSD) solves a per-voxel fiber orientation distribution function (fODF), requiring spherical harmonic decomposition (cuBLAS) at each voxel. Probabilistic tractography (particle filtering, iFOD2) samples millions of streamlines simultaneously, with each streamline step requiring trilinear interpolation of the fODF field — massively parallel across streamlines on GPU. BEDPOSTX GPU accelerates Markov chain Monte Carlo fiber model fitting by 200× vs. CPU.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

DTI (diffusion tensor imaging), NODDI (neurite orientation dispersion), constrained spherical deconvolution (CSD), iFOD2 probabilistic tractography, SIFT/SIFT2 streamline filtering, multi-tissue CSD, particle filtering tractography, deep learning tractography (TractSeg).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/diffusion-mri-tractography.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/diffusion-mri-tractography.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\diffusion-mri-tractography.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Human Connectome Project (HCP) — 1,200 subjects, 3T/7T multi-shell dMRI (https://db.humanconnectome.org/); ABCD Study dMRI (https://abcdstudy.org/); UK Biobank dMRI (https://www.ukbiobank.ac.uk/); TMS-EEG Tractography Contest (verify URL).

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

MRtrix3 (https://github.com/MRtrix3/mrtrix3) — gold-standard CSD, iFOD2, SIFT2, GPU-accelerated deconvolution; FSL BEDPOSTX GPU (https://fsl.fmrib.ox.ac.uk/) — GPU Bayesian fiber orientation estimation (200× speedup); TractSeg (https://github.com/MIC-DKFZ/TractSeg) — direct CNN white-matter tract segmentation; DIPY (https://github.com/dipy/dipy) — Python dMRI analysis with GPU-compatible operations.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuBLAS for spherical harmonic matrix products (CSD); custom CUDA kernel for per-voxel DTI tensor fitting (SVD); CUDA random number generation (cuRAND) for probabilistic streamline sampling; texture memory for fODF field interpolation during tractography. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
