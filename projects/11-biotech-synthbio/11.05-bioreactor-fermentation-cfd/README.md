# 11.5 — Bioreactor & Fermentation CFD

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Biotechnology%2C%20Bioprocess%20%26%20Synthetic%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 11: Biotechnology, Bioprocess & Synthetic Biology · Catalog ID `11.5`
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

Industrial bioreactors exhibit complex turbulent flow, gas-liquid mass transfer (O₂/CO₂), and biological reactions that mutually couple over timescales from milliseconds (bubble coalescence) to hours (cell growth). GPU-accelerated LBM or finite-volume CFD resolves the multi-phase (broth + bubbles) hydrodynamics on meshes with millions of cells, enabling scale-up prediction from bench to 10,000-L fermenters. CFD-metabolic hybrid models link local glucose/O₂ concentrations (from CFD) to spatially-resolved metabolic rates (from flux-balance analysis), identifying gradients that stress industrial cultures. Real-time digital twins combining online sensor data with GPU CFD surrogates enable closed-loop bioreactor control.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Turbulent Navier-Stokes (k-ε / k-ω SST), volume-of-fluid (VOF) gas-liquid interface, population balance model for bubble size distribution, Euler-Euler two-phase flow, lattice-Boltzmann for pore-scale mass transfer, physics-informed neural network surrogate, computational morphology (impeller blade design).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/bioreactor-fermentation-cfd.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/bioreactor-fermentation-cfd.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\bioreactor-fermentation-cfd.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: DECHEMA Bioreactor Flow Dataset — PIV measurements in stirred tanks (verify URL via dechema.de); OpenFOAM BioReactor Tutorial Cases (https://www.openfoam.com/); CHO Fed-Batch Time Course Data (BioNumbers DB, https://bionumbers.hms.harvard.edu/); Zenodo fermentation monitoring datasets (search Zenodo "fed-batch bioreactor").

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

OpenFOAM (https://github.com/OpenFOAM) — gas-liquid bioreactor multiphase solvers (multiphaseEulerFoam) with GPU linear algebra; Palabos (https://gitlab.com/unigespc/palabos) — GPU LBM for porous-media and bubble-column flows; NVIDIA PhysicsNeMo (https://github.com/NVIDIA/physicsnemo) — physics-informed surrogate training for CFD; COBRApy (https://github.com/opencobra/cobrapy) — flux-balance metabolic modeling for CFD coupling.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuSPARSE for pressure-velocity coupling in SIMPLE algorithm, CUDA kernels for VOF interface reconstruction, cuDNN for PINN surrogate inference; pattern: full CFD on GPU with AMG preconditioner → extract local O₂/glucose fields → pass to GPU flux-balance metabolic model → update volumetric reaction terms → iterate time step. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
