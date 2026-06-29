# 14.13 — In Silico Organoid Simulation

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Emerging%2C%20Theoretical%20%26%20Grand--Challenge%20Frontiers-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 14: Emerging, Theoretical & Grand-Challenge Frontiers · Catalog ID `14.13`
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

Organoids — self-organizing 3D stem-cell-derived mini-organs — grow via coupled cell division, differentiation, migration, and mechanical deformation. GPU-accelerated vertex models, cellular Potts models (CPM), and off-lattice agent-based models (ABMs) simulate organoid morphogenesis across thousands to millions of cells. A key bottleneck is computing cell-cell contact forces and sorting energies for CPM (Metropolis Monte Carlo), which are embarrassingly parallel over lattice sites. Virtual tissue simulation from real image data (Frontiers 2024) uses GPU-segmented confocal images to initialize physics-based organoid models, enabling patient-specific drug response prediction for personalized oncology.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Cellular Potts Model (CPM) Metropolis Monte Carlo, vertex model for epithelial mechanics, off-lattice center-based model (CBM), reaction-diffusion morphogen fields (Turing), subcellular element model (SEM), mechanical feedback on gene regulatory network.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/in-silico-organoid-simulation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/in-silico-organoid-simulation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\in-silico-organoid-simulation.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Kaggle Sartorius Cell Instance Segmentation (https://www.kaggle.com/c/sartorius-cell-instance-segmentation); OpenCell — protein localization in live cells (https://opencell.czbiohub.org/); CancerOrganoidDB — organoid drug response (verify URL via Hubrecht Institute); NeurIPS Cell Seg Challenge organoid images (verify URL via Grand Challenge).

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

CompuCell3D (https://compucell3d.org/) — GPU-capable CPM organoid simulation; Morpheus (https://morpheus.gitlab.io/) — GPU cellular Potts + reaction-diffusion; Chaste (https://github.com/Chaste/Chaste) — off-lattice ABM for organoid growth; PhysiCell (https://github.com/MathCancer/PhysiCell) — 3D agent-based multicellular GPU-parallelized simulator.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CUDA checkerboard-parallel Metropolis updates for CPM (even/odd lattice coloring), CUDA reaction-diffusion 3D stencils, cuRAND for Monte Carlo move proposals; pattern: organoid image segmentation → GPU initialization of CPM lattice → parallel Metropolis sweeps (checkerboard coloring avoids conflicts) → reaction-diffusion morphogen update → cell-fate decision → geometry output for imaging comparison. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
