# 14.9 — Multi-Physics Tumor / Treatment Digital Twin

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Emerging%2C%20Theoretical%20%26%20Grand--Challenge%20Frontiers-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 14: Emerging, Theoretical & Grand-Challenge Frontiers · Catalog ID `14.9`
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

A cancer digital twin couples tumor growth (reaction-diffusion PDE for cell density + nutrient + oxygen), mechanical deformation of surrounding tissue (FEM), vascular remodeling (angiogenesis ODE), drug pharmacokinetics (PKPD ODE), immunological response, and radiation damage (LQ model), all personalized from serial multimodal imaging. GPU parallelism tackles the stiff multi-physics coupling: the reaction-diffusion grid (512³ voxels), the FEM mesh (500K elements), and the vascular graph (10⁴ vessel segments) each run on separate GPU streams, synchronized at each time step. Multi-GPU inverse problem fitting of all biophysical parameters to longitudinal MRI + ctDNA data is the frontline computational challenge. The field saw publication of physics-informed ML digital twins for prostate cancer (PSA-driven) in Nature npj Digital Medicine 2025.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Anisotropic tumor-growth reaction-diffusion PDE (Fisher-Kolmogorov), vascular angiogenesis ODE (VEGF-driven), linear-quadratic (LQ) radiation damage model, pharmacokinetic two-compartment model, Bayesian ensemble Kalman filter for parameter assimilation, adjoint-based sensitivity for PDE inversion.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/multi-physics-tumor-treatment-digital-twin.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/multi-physics-tumor-treatment-digital-twin.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\multi-physics-tumor-treatment-digital-twin.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: TCIA (The Cancer Imaging Archive) — multimodal tumor imaging (https://www.cancerimagingarchive.net/); TCGA (The Cancer Genome Atlas) — multi-omics tumor data (https://www.cancer.gov/tcga); ISPY2 — breast cancer treatment response imaging trial (https://www.ispy2.org/); NSCLC-Radiomics (Lung1) — CT + survival on 422 patients (https://www.cancerimagingarchive.net/).

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

CHASTE (https://github.com/Chaste/Chaste) — cancer multiscale + vascular simulation; OpenCMISS-Iron (https://github.com/OpenCMISS/iron) — GPU FEM for tumor-tissue mechanics; NVIDIA PhysicsNeMo (https://github.com/NVIDIA/physicsnemo) — PINN surrogates for tumor growth; TumorFEM (verify URL, search "tumor digital twin FEM GitHub") — patient-specific tumor mechanical FEM.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CUDA 3D stencil kernels for reaction-diffusion PDE, cuSPARSE for FEM tissue mechanics, cuSolver for vascular pressure-flow network, multi-GPU NCCL for coupled physics domains; pattern: patient MRI → tumor/tissue segmentation → multi-physics GPU simulation → synthetic MRI generation → Bayesian parameter assimilation → treatment prediction. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
