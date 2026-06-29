# 13.5 — In Silico Virtual Clinical Trials

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Pharmacology%20%26%20Clinical%20Quantitative%20Modeling-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 13: Pharmacology & Clinical Quantitative Modeling · Catalog ID `13.5`
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

Generates virtual patient populations and runs complete simulated clinical trials in silico to optimise dose, schedule, and eligibility criteria before committing to expensive Phase II/III studies. Each virtual patient is characterised by a parameter set sampled from a PBPK/PD population distribution; simulating 5000 virtual patients through 24-week dose schedules requires 5000 independent ODE trajectories, each with ~50 compartments and hundreds of time steps. GPU-parallel batched ODE integration reduces trial simulation time from hours to seconds. Optimal virtual trial design uses GPU-resident Bayesian optimisation over dose/schedule space.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Monte Carlo virtual population generation, population PBPK/PD ODE integration, Latin hypercube sampling of parameter space, Bayesian optimisation of trial design parameters (dose, schedule, N), survival analysis on simulated endpoints, regulatory-grade power calculation, sensitivity analysis (Morris screening, Sobol indices).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/in-silico-virtual-clinical-trials.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/in-silico-virtual-clinical-trials.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\in-silico-virtual-clinical-trials.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Open Systems Pharmacology virtual patient databases (https://github.com/Open-Systems-Pharmacology/) ClinicalTrials.gov schema — trial design parameters for calibration (https://clinicaltrials.gov/) FDA CDER pharmacometric review datasets (verify URL via FDA) Published dose-finding trial datasets in CDISC format (https://www.cdisc.org/)

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

Pumas (https://pumas.ai/) — GPU-accelerated virtual clinical trials in Julia nvQSP (https://github.com/NVIDIA-Digital-Bio/nvQSP) — GPU PBPK ODE solver for virtual patient simulation SimBiology (MATLAB Parallel Computing Toolbox) — virtual trial simulation with cluster/GPU backend (verify URL) PKPD Simulator (verify URL) — open Python framework for virtual trial simulation

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CUDA batched RK45 for thousands of simultaneous patient ODE trajectories, cuRAND for virtual population parameter sampling, Thrust for summary statistic aggregation; pattern: SIMD-parallel ODE integration with each virtual patient in a CUDA warp. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
