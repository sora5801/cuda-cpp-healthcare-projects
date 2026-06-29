# 1.24 — Umbrella Sampling / WHAM Free Energy Profiles

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟢 Beginner · Established** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.24`
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

Umbrella sampling applies harmonic restraints along a reaction coordinate (e.g., ligand unbinding distance, pore radius) to force sampling at energy barriers. Multiple windows run simultaneously (embarrassingly parallel across GPUs), each an independent GPU MD simulation. WHAM (Weighted Histogram Analysis Method) or MBAR post-processes window histograms into a potential of mean force (PMF). GPU MD enables each window to generate nanoseconds of biased trajectory in minutes, enabling convergence that was previously impractical. Applications include permeation barriers in ion channels and drug binding/unbinding free energy profiles.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Harmonic bias potentials, WHAM self-consistent iteration, MBAR (multistate BAR), steered MD + Jarzynski equality, metadynamics PMF (alternative), local elevation/flooding.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/umbrella-sampling-wham-free-energy-profiles.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/umbrella-sampling-wham-free-energy-profiles.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\umbrella-sampling-wham-free-energy-profiles.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Ion channel permeation benchmark sets; SAMPL binding free energy challenges (https://github.com/samplchallenges/SAMPL); BindingDB (https://www.bindingdb.org); GROMACS umbrella sampling tutorials (https://tutorials.gromacs.org).

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

GROMACS gmx wham (https://github.com/gromacs/gromacs) — built-in WHAM post-processing; OpenMM umbrella sampling (https://github.com/openmm/openmm-cookbook) — Python harmonic restraints; alchemlyb (https://github.com/alchemistry/alchemlyb) — MBAR/WHAM post-processing; PLUMED (https://github.com/plumed/plumed2) — collective variables + restraints.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Full MD per window on GPU; MPI + NCCL to launch window array; WHAM iteration on CPU via numpy; GPU-parallel histogram accumulation using atomicAdd; shared-memory reductions for collective variable forces. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
