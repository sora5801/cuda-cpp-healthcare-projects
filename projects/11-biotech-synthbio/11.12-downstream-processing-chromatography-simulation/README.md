# 11.12 — Downstream Processing & Chromatography Simulation

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Biotechnology%2C%20Bioprocess%20%26%20Synthetic%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 11: Biotechnology, Bioprocess & Synthetic Biology · Catalog ID `11.12`
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

Protein A affinity, ion-exchange, and size-exclusion chromatography columns for antibody purification are governed by advection-dispersion-reaction (ADR) PDEs coupled with adsorption isotherm equations (steric mass action, SMA). GPU-accelerated PDE solvers (finite-volume or spectral methods) simulate full column dynamics in seconds per run, enabling in silico process characterization (DoE) across 100s of loading, wash, and elution conditions in parallel. Inverse problem fitting of SMA parameters from batch isotherm experiments uses GPU-accelerated Bayesian optimization. The bottleneck is the large stiff ODE system for multi-component competitive adsorption.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Advection-dispersion-reaction PDE (Godunov scheme / WENO), steric mass action (SMA) isotherm model, general rate model (GRM), shrinking core diffusion model, Bayesian optimization for process development, GPU-parallel Latin hypercube DoE.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/downstream-processing-chromatography-simulation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/downstream-processing-chromatography-simulation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\downstream-processing-chromatography-simulation.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: CADET Benchmark Cases — chromatography simulation validation (https://github.com/modsim/CADET); USP Bioprocess Data Repository — chromatography process development records (verify URL via NIST/USP); PDB-based antibody charge maps for adsorption prediction; OpenChrom mass-spectrometry chromatography datasets (https://www.openchrom.net/).

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

CADET (https://github.com/modsim/CADET) — Chromatography Analysis and Design Toolkit, CPU reference; CADET-Process (https://github.com/modsim/CADET-Process) — Python optimization wrapper for CADET; GPU-ADR solvers via CUDA finite-volume (custom implementation, verify via GitHub search "GPU chromatography simulation"); PyTorch surrogate for chromatography (verify URL via Biotechnology Journal 2024).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CUDA finite-volume kernels for 1D PDE time-stepping (one thread per spatial grid point), cuSPARSE for implicit diffusion system, Thrust for parallel DoE condition enumeration; pattern: 200 chromatography conditions enumerated → GPU PDE solve per condition in parallel → elution profile extraction → Bayesian optimizer selects next DoE → iterate until convergence. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
