# 6.8 — Tumor Growth & Treatment-Response Modeling

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.8`
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

Continuum-PDE models (reaction-diffusion for nutrient/oxygen, tumor cell density, and treatment drug concentration) combined with discrete cell-based models capture avascular-to-vascular tumor growth, hypoxia-driven necrosis, and response to radiation or chemotherapy. GPU acceleration is essential for solving coupled PDE systems on 3D grids (512³ voxels = 1.3×10⁸ cells) at each time step of a multi-day simulation. Parameter sweeps for virtual clinical trials (thousands of parameter sets) are embarrassingly parallel across the GPU grid.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Fisher-KPP reaction-diffusion (tumor cell density), oxygen/nutrient diffusion-consumption (Green's function or FD), phenomenological radiobiological model (linear-quadratic), drug PK/PD compartment coupling, phase-field tumor morphology, level-set interface tracking for tumor boundary.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/tumor-growth-treatment-response-modeling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/tumor-growth-treatment-response-modeling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\tumor-growth-treatment-response-modeling.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: TCGA (The Cancer Genome Atlas) — multi-omics + imaging for model calibration (https://portal.gdc.cancer.gov); TCIA (The Cancer Imaging Archive) — multi-institutional tumor imaging (https://www.cancerimagingarchive.net); PhysioNet oncology waveforms (https://physionet.org); Zenodo tumor growth simulation datasets (search zenodo.org for "tumor growth simulation").

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

PhysiCell (https://github.com/MathCancer/PhysiCell) — 3D agent-based multicellular simulator with diffusing substrates, scales linearly in cell count; PhysiBoSS (https://github.com/PhysiBoSS/PhysiBoSS) — extends PhysiCell with Boolean network intracellular signaling (MaBoSS); Chaste (https://github.com/Chaste/Chaste) — includes tumor spheroid and crypt models; OpenFOAM (https://github.com/OpenFOAM/OpenFOAM-dev) — used for drug delivery flow simulations.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom CUDA FD stencil kernels (3D 7-point Laplacian on oxygen/drug grids), CUDA Thrust for per-cell agent sorting and binning, cuRAND for stochastic division/death events; pattern: 3D CUDA thread grid for PDE, separate kernel for agent-based cell loop with shared-memory neighborhood queries. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
