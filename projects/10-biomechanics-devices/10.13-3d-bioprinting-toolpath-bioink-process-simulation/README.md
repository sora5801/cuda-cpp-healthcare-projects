# 10.13 — 3D Bioprinting Toolpath & Bioink Process Simulation

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Biomechanics%2C%20Biomedical%20Devices%20%26%20Surgery-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 10: Biomechanics, Biomedical Devices & Surgery · Catalog ID `10.13`
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

Extrusion-based bioprinting deposits cell-laden hydrogels through a nozzle, where shear stress during extrusion determines post-print cell viability. GPU-accelerated CFD of the nozzle + deposition region (non-Newtonian Carreau fluid) predicts wall shear stress as a function of nozzle geometry, ink rheology, and print speed, enabling parameter optimization in silico before costly biological experiments. Lattice-structure scaffold design — maximizing permeability for nutrient transport while maintaining mechanical stiffness — uses GPU topology optimization with fluid-flow homogenization. Thermal modeling of photopolymerization in DLP/SLA bioprinting on GPU resolves crosslink-front propagation in real time.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Non-Newtonian Navier-Stokes (Carreau-Yasuda viscosity model), topology optimization with permeability (Darcy-Stokes coupling), heat-transfer / photo-crosslinking kinetics, support-structure generation via GPU ray casting, ML surrogate (XGBoost/MLP) for viability prediction.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/3d-bioprinting-toolpath-bioink-process-simulation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/3d-bioprinting-toolpath-bioink-process-simulation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\3d-bioprinting-toolpath-bioink-process-simulation.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: In silico Bioink Viability Dataset (Zenodo) — extrusion viability vs. shear-stress features (https://zenodo.org/records/11545357); BioInk Rheology Database (verify URL via Biofabrication journal); 3D Bioprinting Benchmarks (verify URL via Zenodo); Scaffold Permeability Benchmark (https://arxiv.org/abs/1104.1028).

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

in-silico-bioink-viability-prediction (https://github.com/KORINZ/in-silico-bioink-viability-prediction) — ML viability prediction from shear stress; OpenFOAM (https://github.com/OpenFOAM) — non-Newtonian flow solver for nozzle CFD; FEBio (https://github.com/febiosoftware/FEBio) — scaffold mechanical FEA; TPMS Scaffold Generator (verify URL via GitHub) — GPU-accelerated triply-periodic-minimal-surface lattice generation.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CUDA kernels for non-Newtonian viscosity update per cell, cuFFT for spectral pressure solve, cuDNN for surrogate viability model inference; pattern: parametric nozzle geometry → GPU Navier-Stokes solve for shear-stress field → shear-stress statistics fed to GPU ML surrogate → output: print parameters vs. predicted viability Pareto front. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
