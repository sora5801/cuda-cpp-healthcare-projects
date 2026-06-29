# 2.28 — Replica Exchange Solute Tempering (REST2) on GPU

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.28`
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

REST2 (Replica Exchange with Solute Tempering version 2) selectively heats only the solute (protein/ligand) degrees of freedom rather than the whole system, making replica exchange practical for large solvated systems where heating all water would be prohibitively expensive. Effective temperature scaling is applied only to protein internal and protein-water interactions, while water-water interactions remain at 300K. GPU MD runs each replica independently; NCCL/MPI handles exchange coordinate communication between replicas at swap intervals. Applications include enhanced sampling of protein-ligand binding, loop conformational changes, and protein folding.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Scaled Hamiltonian construction for solute interactions, Metropolis exchange criterion across replicas, potential energy re-scaling (protein-protein + protein-solvent), REST2 vs HREX comparison, virtual replica exchange (vRE-REST2).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/replica-exchange-solute-tempering-rest2-on-gpu.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/replica-exchange-solute-tempering-rest2-on-gpu.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\replica-exchange-solute-tempering-rest2-on-gpu.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Shaw millisecond folding trajectories for validation; SAMPL challenges (https://github.com/samplchallenges/SAMPL); GPCRmd REST2 enhanced sampling data (https://gpcrmd.org); chignolin/Trp-cage fast-folder benchmarks.

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

GROMACS + PLUMED REST2 (https://github.com/gromacs/gromacs) — Hamiltonian REMD on GPU; NAMD REST2 (https://www.ks.uiuc.edu/Research/namd/) — GPU replica exchange; OpenMM REST2 via openmmtools (https://github.com/choderalab/openmmtools) — Python REST2 on GPU; DESMOND REST2 (Schrodinger, commercial) — GPU REST2 for FEP.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Independent GPU MD per replica; NCCL for energy exchange between GPUs; CUDA Hamiltonian scaling kernel (scale protein-water pair forces); MPI inter-node replica exchange; GPU-parallel Metropolis criterion evaluation. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
