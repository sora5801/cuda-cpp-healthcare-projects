# 13.11 — Bayesian PK/PD Inference on GPU (Torsten/Stan)

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Pharmacology%20%26%20Clinical%20Quantitative%20Modeling-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 13: Pharmacology & Clinical Quantitative Modeling · Catalog ID `13.11`
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

Full Bayesian inference for PK/PD models using Hamiltonian Monte Carlo (HMC/NUTS) within Stan + Torsten, where the log-posterior gradient requires integrating population ODE trajectories and evaluating the likelihood. Each HMC leapfrog step requires one full ODE solve per patient in the dataset — for 1000 patients × 2000 HMC iterations × 10 leapfrog steps = 20M ODE solves per chain. GPU acceleration of these batched ODE solves provides the critical speedup. The `reduce_sum` function in Stan enables within-chain parallelism across patients on multi-core CPU; true GPU acceleration requires the CUDA ODE integration backends available through Pumas or experimental Stan GPU interfaces.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Hamiltonian Monte Carlo (HMC), No-U-Turn Sampler (NUTS), automatic differentiation through ODE solvers (adjoint sensitivity), Runge-Kutta ODE integration, adaptive dual-averaging stepsize, Bayesian predictive check (PPC), R-hat convergence diagnostics, Bayesian cross-validation (LOO-CV).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/bayesian-pk-pd-inference-on-gpu-torsten-stan.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/bayesian-pk-pd-inference-on-gpu-torsten-stan.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\bayesian-pk-pd-inference-on-gpu-torsten-stan.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Torsten example models (https://github.com/metrumresearchgroup/Torsten) — 2-compartment, PKPD, TMDD Stan models Somatrogon population PK dataset (ResearchGate, 2024) — Bayesian NLME application with Torsten Warfarin PK/PD dataset — standard Bayesian NLME benchmark (verify URL) MIMIC-IV medication + lab values — vancomycin TDM for Bayesian dosing (https://physionet.org/content/mimiciv/)

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

Torsten (https://github.com/metrumresearchgroup/Torsten) — Stan ODE extensions for PK/PD; SAEM and HMC CmdStanR / CmdStanPy (https://mc-stan.org/cmdstanr/) — Stan interface for running GPU-parallel chains Pumas (https://pumas.ai/) — Julia Bayesian PK/PD with GPU-accelerated HMC via CUDA.jl MCMCChains (https://github.com/TuringLang/MCMCChains.jl) — MCMC diagnostics for population PK posteriors

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

GPU-parallelised ODE solvers called from Stan adjoint sensitivity method, cuBLAS for Hessian approximation, NCCL for multi-chain parallelism; pattern: multi-GPU chains run in parallel with NCCL synchronisation for diagnostics. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
