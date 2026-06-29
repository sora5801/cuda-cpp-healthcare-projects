# 6.5 — Respiratory / Lung Airflow & Particle Deposition

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.5`
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

Simulates inspiratory/expiratory flow through the conducting airways (generations 0–16, reconstructed from CT) and tracks inhaled aerosol/drug particle trajectories via Lagrangian particle tracking. The lung's tree topology means ~10⁶–10⁷ computational cells in the airway geometry and millions of particle trajectories evaluated each breath cycle—both trivially parallelizable on GPU. Alveolar gas exchange adds a reaction-diffusion layer for O₂/CO₂ that couples to a 1D ventilation model for the respiratory tree periphery.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Incompressible Navier-Stokes (finite volume), Lagrangian discrete-phase particle tracking (drag + Brownian + Saffman lift forces), Stokes drag law, k-ω SST RANS turbulence, LBM for alveolar-scale flow, convection-diffusion for gas species, quasi-1D ventilation model (Horsfield tree).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/respiratory-lung-airflow-particle-deposition.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/respiratory-lung-airflow-particle-deposition.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\respiratory-lung-airflow-particle-deposition.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: LIDC-IDRI lung CT — 1 010 cases with nodule annotations, TCIA (https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI); COPDGene lung CT dataset — 10 000 subjects (https://www.copdgene.org); SPIROMICS bronchial CT (https://www.spiromics.org); PhysioNet respiratory waveform databases (https://physionet.org).

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

OpenFOAM-dev (https://github.com/OpenFOAM/OpenFOAM-dev) — Lagrangian particle tracking (DPMFoam) with GPU-capable solver via GPU-accelerated AmgX pressure solve; SimVascular (https://github.com/SimVascular) — vascular flow basis adaptable to airways; PALABOS (https://gitlab.com/unigespc/palabos) — LBM for alveolar flow; 3D Slicer + SlicerMorph (https://github.com/SlicerMorph/SlicerMorph) — airway segmentation from CT.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CUDA Thrust for particle sort/bin operations; custom CUDA kernels for Lagrangian force integration (one thread per particle); cuSPARSE for airflow linear solve; pattern: dual-stream approach—Eulerian fluid on one SM partition, Lagrangian particles on another with atomic-add deposition counters. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
