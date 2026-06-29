# 4.27 — Radiomics Feature Extraction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.27`
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

Radiomics extracts hundreds of quantitative features (shape, first-order statistics, texture: GLCM, GLRLM, GLSZM, NGTDM) from 3D segmented ROIs in CT/PET/MRI. For a cohort of 10,000 patients with large ROIs (~10⁶ voxels each), CPU-based PyRadiomics takes 10–30 min per patient; GPU-accelerated cuRadiomics and PyRadiomics-CUDA achieve 143× speedup by parallelizing all histogram and co-occurrence matrix computations across voxels on GPU. Texture features require computing co-occurrence matrices from 26 3D neighbor directions simultaneously — each direction's computation is independent, enabling massive GPU parallelism. Radiomics biomarker discovery pipelines must process thousands of scans for statistical power.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

GLCM (gray-level co-occurrence matrix), GLRLM (run-length matrix), GLSZM (size-zone matrix), NGTDM (neighborhood gray-tone difference matrix), first-order statistics, 3D shape descriptors, wavelet-decomposition features, multi-scale radiomics, IBSI (Image Biomarker Standardization Initiative) compliant features.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/radiomics-feature-extraction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/radiomics-feature-extraction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\radiomics-feature-extraction.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: TCIA NSCLC-Radiomics (https://www.cancerimagingarchive.net/collection/nsclc-radiomics/) — 422 lung CTs with survival; RIDER Breast MRI (via TCIA); QIN-HEADNECK (via TCIA) — head and neck RT; TCGA collections (https://portal.gdc.cancer.gov/).

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

PyRadiomics-CUDA (https://arxiv.org/abs/2510.02894 — code on https://github.com/mis-wut/pyradiomics-CUDA) — GPU radiomics, 143× speedup; cuRadiomics (verify URL — published in AAPM proceedings) — CUDA texture/GLCM GPU extraction; PyRadiomics CPU baseline (https://github.com/AIM-Harvard/pyradiomics) — IBSI-compliant reference; MONAI (https://github.com/Project-MONAI/MONAI) — integrated GPU radiomics pipeline.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom CUDA for co-occurrence matrix (atomic add into per-direction GLCM per thread block); shared memory for voxel neighborhood; parallel histogram across all voxels (CUB block histogram); warp-level reductions for matrix statistics. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
