# 1.3 — Molecular Docking Engine

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟢 Beginner · Established** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.3`
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

Molecular docking predicts the preferred binding pose and score of a small molecule within a protein binding pocket by sampling ligand conformations (translations, rotations, torsions) and scoring each with an empirical or knowledge-based energy function. The scoring function evaluation for each sampled pose is independent, creating massive data parallelism — thousands of poses per ligand, millions of ligands per campaign. AutoDock-GPU achieves >1000× speedup over single-CPU AutoDock4 by running the Lamarckian genetic algorithm (LGA) in parallel across GPU threads, each evaluating a distinct pose. The bottleneck is the grid-based force-field energy lookup, which benefits from GPU texture-cache acceleration.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Lamarckian Genetic Algorithm (LGA), gradient-based local search (BFGS), grid-based energy evaluation (electrostatics + vdW precalculated on 3D grids), scoring functions (AutoDock4, Vina, Vinardo, AD4).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/molecular-docking-engine.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/molecular-docking-engine.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\molecular-docking-engine.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: DUD-E — directory of useful decoys enhanced, 102 targets with actives and decoys (https://dude.docking.org); ChEMBL — bioactivity database with >2M compounds (https://www.ebi.ac.uk/chembl/); PDB-bind — curated protein-ligand complexes with binding affinities (http://www.pdbbind.org.cn); CASF benchmark — comparative assessment of scoring functions (http://www.pdbbind.org.cn/casf.php).

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

AutoDock-GPU (https://github.com/ccsb-scripps/AutoDock-GPU) — CUDA/OpenCL GPU docking with LGA parallelism; Uni-Dock (https://github.com/dptech-corp/Uni-Dock) — GPU-accelerated batch docking with >2000× speedup on V100; GNINA (https://github.com/gnina/gnina) — CNN-scored docking fork of smina; Vina-GPU 2.1 (https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1) — GPU-accelerated AutoDock Vina with RILC-BFGS.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Texture memory for 3D grid lookups, CUDA threadblocks each running one GA individual per ligand, warp-level reduction for fitness evaluation; grid-strided loops for pose batch processing. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
