# 4.14 — Digital Breast Tomosynthesis

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.14`
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

Digital breast tomosynthesis (DBT) acquires 9–25 low-dose projections over a limited angular range (~15–50°), then reconstructs thin slabs through compressed breast tissue. The limited-angle geometry makes analytical FBP unstable, so iterative methods (OS-EM, SART, ASD-POCS) with total-variation regularization dominate for artifact reduction. The breast is a low-contrast, soft-tissue object where noise and blur from the limited angle severely reduce lesion conspicuity, making statistical reconstruction critical. A single DBT volume (~800 × 700 × 60 slices at 85 µm) represents ~30 GB of raw projection data; GPU acceleration reduces OS-EM reconstruction from hours to under a minute. Deep learning methods (U-Net denoising on FBP outputs) additionally require GPU for inference.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

FBP with limited-angle filter, OS-EM (ordered-subsets EM), SART, ASD-POCS with total variation, model-based iterative reconstruction (MBIR), DBT-specific PSF/MTF modelling, deep learning denoising and artifact reduction, mass detection CNNs.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/digital-breast-tomosynthesis.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/digital-breast-tomosynthesis.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\digital-breast-tomosynthesis.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: OPTIMAM Mammography Image Database (OMI-DB, access via ICR UK); CBIS-DDSM (https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM) — 2,620 mammograms via TCIA; VinDr-Mammo (https://physionet.org/content/vindr-mammo/1.0.0/); BCS-DBT (Duke DBT challenge dataset, https://bcs-dbt.grand-challenge.org/).

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

ASTRA Toolbox (https://github.com/astra-toolbox/astra-toolbox) — GPU forward/back-projection for arbitrary cone-beam geometry; RTK (https://github.com/RTKConsortium/RTK) — FDK and iterative DBT-capable; TIGRE (https://github.com/CERN/TIGRE) — DBT-compatible geometry; OpenDBT (verify URL) — research-focused DBT reconstruction framework.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuFFT for ramp filter; CUDA voxel-driven backprojection with compressed breast geometry; texture memory for projection interpolation; limited-angle geometry stored in constant memory; ADMM inner loop GPU-resident. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
