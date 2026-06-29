# 2.9 — Solvent-Accessible Surface & Poisson-Boltzmann Electrostatics

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟢 Beginner · Established** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.9`
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

Continuum electrostatics models (Poisson-Boltzmann equation, PBE) compute the electrostatic potential of a protein in ionic solvent by solving a partial differential equation on a 3D grid. This enables calculation of protein pKa values, electrostatic binding contributions, and zeta potentials for colloidal drug carriers. GPU-accelerated PBE solvers (APBS, DelPhi-GPU) discretize the molecule onto a Eulerian grid and solve via Gauss-Seidel iteration or multigrid methods on GPU. The bottleneck is the 3D finite-difference PBE solve — parallelized via coloring (red-black ordering) on GPU threads.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Linearized Poisson-Boltzmann equation (LPBE), non-linear PBE, finite difference discretization (3D grid), red-black Gauss-Seidel iteration, multigrid preconditioning, generalized Born (GB) analytic approximation, SASA computation.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/solvent-accessible-surface-poisson-boltzmann-electrostatics.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/solvent-accessible-surface-poisson-boltzmann-electrostatics.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\solvent-accessible-surface-poisson-boltzmann-electrostatics.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: pKDBD — database of protein pKa values (verify URL); BindingMOAD — protein-ligand electrostatic data (https://bindingmoad.org); RCSB PDB structural data (https://www.rcsb.org); APBS validation benchmark (https://github.com/Electrostatics/apbs).

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

APBS (https://github.com/Electrostatics/apbs) — Poisson-Boltzmann solver with GPU acceleration; DelPhi (http://compbio.clemson.edu/delphi) — PB electrostatics with GPU solver; OpenMM GB force (https://github.com/openmm/openmm) — GPU Generalized Born; PDB2PQR (https://github.com/Electrostatics/pdb2pqr) — structure preparation for PBE.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CUDA thread blocks for 3D finite-difference red-black iteration; shared memory for stencil computation; cuSPARSE for sparse Laplacian matrix; GPU texture memory for dielectric boundary representation. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
