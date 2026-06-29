# 6.10 — Systems-Biology ODE/SDE Network Solver

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.10`
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

Gene regulatory networks, signaling cascades, and metabolic models are encoded as systems of potentially thousands of nonlinear ODEs/SDEs (e.g., SBML models from BioModels). Integrating a single model is fast, but parameter sweeps, uncertainty quantification, and multi-cell applications require solving thousands of independent instances simultaneously—a perfectly GPU-parallel batch problem. SUNDIALS/CVODE-GPU and libRoadRunner's LLVM JIT backend both target this batch-ODE pattern.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

CVODE adaptive BDF/Adams multistep integrator, explicit Euler / Runge-Kutta (RK4, RK45 Dormand-Prince) for stiff-moderate systems, implicit trapezoidal, chemical Langevin equation (CLE) for SDE, sensitivity equations (CVODES/IDAS), SBML parsing and JIT compilation.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/systems-biology-ode-sde-network-solver.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/systems-biology-ode-sde-network-solver.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\systems-biology-ode-sde-network-solver.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: BioModels Database (EMBL-EBI) — 1000+ curated SBML models (https://www.ebi.ac.uk/biomodels); Reactome pathways — curated molecular interaction data (https://reactome.org); BioGRID interaction network (https://thebiogrid.org); VCell curated models (https://vcell.org).

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

SUNDIALS/CVODE GPU (https://github.com/LLNL/sundials) — LLNL ODE/DAE solver with CUDA NVector and GPU-accelerated batch CVODE; libRoadRunner (https://github.com/sys-bio/roadrunner) — high-performance SBML ODE integrator with LLVM JIT, GPU batch mode in development; Tellurium (https://github.com/sys-bio/tellurium) — Python systems biology platform built on roadrunner; GillesPy2 (https://github.com/GillesPy2/GillesPy2) — SSA + tau-leaping + CLE stochastic solver.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CUDA batched ODE: one CUDA thread-block per ODE system; shared memory for Jacobian; cuSPARSE for large sparse Jacobians; SUNDIALS CUDA NVector; pattern: batch-CVODE with user-supplied CUDA right-hand-side (RHS) kernel. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
