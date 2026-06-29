# 2.26 — Hydrogen Bond Network & Water Placement Analysis

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.26`
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

Water molecules mediate protein-ligand interactions at binding sites; their correct placement is critical for accurate docking and scoring. GPU-accelerated MD generates explicit water trajectories from which statistical water occupancy maps (WaterMap, GIST) are computed. The Grid Inhomogeneous Solvation Theory (GIST) requires computing per-voxel thermodynamic quantities (energy, entropy) across millions of trajectory frames — a GPU-parallelizable grid accumulation problem. High-occupancy waters indicate entropically costly displacement sites; displacing them with ligand atoms typically yields affinity gains.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Grid Inhomogeneous Solvation Theory (GIST), Inhomogeneous Fluid Solvation Theory (IFST), 3D water occupancy map from MD, nearest-neighbor entropy estimation, water bridge H-bond network graph, explicit water clustering.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/hydrogen-bond-network-water-placement-analysis.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/hydrogen-bond-network-water-placement-analysis.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\hydrogen-bond-network-water-placement-analysis.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: SAMPL water placement challenges (https://github.com/samplchallenges/SAMPL); explicit-solvent PDB structures (https://www.rcsb.org); benchmark sets for WaterMap validation (Schrodinger, verify URL); GIST reference calculations for T4 lysozyme and FKBP12.

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

GISTPP (https://github.com/liedlgroup/gist-pp) — GIST water thermodynamics analysis; cpptraj GIST (https://github.com/Amber-MD/cpptraj) — AMBER trajectory analysis with GIST; MDAnalysis water analysis (https://github.com/MDAnalysis/mdanalysis) — H-bond and water bridge analysis; WaterMD (verify URL) — GPU-accelerated solvation free energy.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

GPU grid accumulation kernels for GIST voxel energy/entropy (atomic updates); custom CUDA nearest-neighbor entropy estimation; MDAnalysis GPU trajectory streaming; GPU-parallel water oxygen occupancy histogramming. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
