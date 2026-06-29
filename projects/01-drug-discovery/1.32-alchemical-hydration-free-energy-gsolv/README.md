# 1.32 — Alchemical Hydration Free Energy (ΔGsolv)

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.32`
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

Absolute solvation free energies (ΔGhyd for water, ΔGsolv for organic solvents) are foundational to ADMET modeling (LogP, LogS, membrane permeability). Alchemical calculation via thermodynamic integration or FEP decouples solute-solvent interactions over λ-windows, yielding ΔGsolv directly from GPU MD simulations. Compared to QSAR models, GPU alchemical ΔGsolv achieves sub-kcal/mol accuracy on drug-like molecules. FreeSolv benchmark provides 643 experimental hydration free energies for validation.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Alchemical decoupling (electrostatics then LJ), soft-core potentials, MBAR/TI post-processing, absolute binding free energy (ABFE), double decoupling, Bennet acceptance ratio (BAR).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/alchemical-hydration-free-energy-gsolv.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/alchemical-hydration-free-energy-gsolv.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\alchemical-hydration-free-energy-gsolv.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: FreeSolv — 643 experimental hydration free energies (https://github.com/MobleyLab/FreeSolv); MNSol — Minnesota solvation database (https://comp.chem.umn.edu/mnsol/); SAMPL hydration challenges (https://github.com/samplchallenges/SAMPL); NIST ThermoML hydration data (https://trc.nist.gov).

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

OpenFE (https://github.com/OpenFreeEnergy/openfe) — open alchemical FE toolkit; alchemtest (https://github.com/alchemistry/alchemtest) — test systems for alchemical codes; GROMACS + alchemlyb (https://github.com/gromacs/gromacs) — GPU FEP pipeline; AMBER FEP (https://ambermd.org) — pmemd.cuda alchemical decoupling.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Full GPU MD (cuFFT PME + custom force kernels); parallel λ-window MD across GPU array; MBAR post-processing via pymbar; GPU evaluation of soft-core potential perturbations. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
