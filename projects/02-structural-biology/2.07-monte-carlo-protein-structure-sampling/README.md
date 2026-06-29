# 2.7 — Monte Carlo Protein Structure Sampling

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.7`
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

Monte Carlo (MC) methods sample protein conformational space by proposing random moves (backbone/sidechain dihedral rotations, rigid-body domain motions) and accepting/rejecting via Metropolis criterion. GPU acceleration is applied to (i) batch scoring of many independent MC walkers in parallel and (ii) GPU-accelerated energy evaluation for each trial move. Rosetta's protein design/folding MC engine has been partially GPU-accelerated. Parallel tempering MC scales to GPU arrays via independent temperature replicas. Applications include loop modeling, sidechain packing, and protein-ligand pose sampling.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Metropolis-Hastings MC, parallel tempering, fragment-based backbone moves (Rosetta), rotamer library sidechain packing (Dunbrack), basin hopping, simulated annealing, energy function evaluation (Rosetta or AMBER).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/monte-carlo-protein-structure-sampling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/monte-carlo-protein-structure-sampling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\monte-carlo-protein-structure-sampling.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: CASP protein structure benchmarks (https://predictioncenter.org); PDB structures for folding benchmarks (https://www.rcsb.org); Dunbrack rotamer library (https://dunbrack.fccc.edu/bbdep2010/); CAMEO continuous benchmarking (https://www.cameo3d.org).

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

Rosetta (https://github.com/RosettaCommons/rosetta) — protein MC sampling (GPU extensions experimental); FoldX (https://foldxsuite.crg.eu) — fast energy evaluation for MC design; OpenMM MC (https://github.com/openmm/openmm) — Python MC on GPU via custom integrators; ProteinMPNN (https://github.com/dauparas/ProteinMPNN) — GPU sequence design complementary to MC backbone sampling.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

GPU-parallel scoring of independent MC replica arrays; CUDA kernels for energy evaluation (Lennard-Jones + torsion); cuRAND for GPU random number generation; warp-level acceptance ratio evaluation. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
