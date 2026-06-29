# 11.4 — Synthetic-Biology Genetic-Circuit Design & Simulation

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Biotechnology%2C%20Bioprocess%20%26%20Synthetic%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 11: Biotechnology, Bioprocess & Synthetic Biology · Catalog ID `11.4`
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

Genetic-circuit design requires stochastic simulation (Gillespie SSA) of regulatory networks with hundreds of species and reactions, then optimization of promoter strengths, RBS sequences, and protein copy numbers to achieve target transfer-function shapes. GPU parallelism runs thousands of independent SSA trajectories simultaneously on a single card — each trajectory is a separate CUDA stream — reducing Monte Carlo ensemble variance estimation from hours to seconds. Deterministic ODE simulation (Hill kinetics) of large gene regulatory networks (GRNs) further benefits from GPU batch-ODE solvers (cuSolver + custom RK4). Bayesian optimization over the genetic parameter space closes the design loop.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Gillespie Stochastic Simulation Algorithm (SSA), tau-leaping (accelerated SSA), deterministic ODE integration with Hill-function kinetics, Bayesian optimization (GP-UCB) for parameter tuning, coarse-grained thermodynamic models for promoter strength, Boolean logic gate composition.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/synthetic-biology-genetic-circuit-design-simulation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/synthetic-biology-genetic-circuit-design-simulation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\synthetic-biology-genetic-circuit-design-simulation.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: iGEM Registry of Standard Biological Parts — promoter/RBS/gene part catalog (https://parts.igem.org/); SBOL Designer parts library (https://sboldesigner.github.io/); BioBrick Characterization Database (verify URL via SynBioHub); Promoter Strength Library (Anderson promoter series) (verify URL via parts.igem.org).

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

Tellurium (https://github.com/sys-bio/tellurium) — Python ODE/SSA simulator for SBML models with CUDA-extensible solvers; GillesPy2 (https://github.com/StochSS/GillesPy2) — Python SSA with GPU acceleration roadmap; COPASI (https://github.com/copasi/COPASI) — biochemical network simulator with parallel parameter scanning; iBioSim (https://github.com/MyersResearchGroup/iBioSim) — genetic circuit design + simulation framework.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CUDA kernels for parallel SSA trajectories (one trajectory per thread block), cuRAND for per-trajectory random number streams, cuSolver for stiff ODE Jacobian factorization; pattern: genetic circuit model → 10⁴ GPU SSA trajectories in parallel → histogram-based transfer-function estimation → Bayesian optimizer proposes new promoter parameters → iterate. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
