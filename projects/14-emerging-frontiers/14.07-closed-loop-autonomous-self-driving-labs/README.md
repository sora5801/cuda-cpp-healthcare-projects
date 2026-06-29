# 14.7 — Closed-Loop Autonomous "Self-Driving" Labs

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Emerging%2C%20Theoretical%20%26%20Grand--Challenge%20Frontiers-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 14: Emerging, Theoretical & Grand-Challenge Frontiers · Catalog ID `14.7`
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

Self-driving labs (SDLs) close the design-build-test-learn cycle by coupling GPU-accelerated Bayesian optimization (BO) or reinforcement learning to robotic liquid handlers, automated assays, and real-time data pipelines. The GPU role is the inner-loop inference: scoring thousands of candidate experiments via surrogate models (GP, neural network ensembles) in milliseconds, so the acquisition function evaluates faster than the robot can dispense. Active learning for drug discovery (e.g., Gaussian Process + batch BO with qEI) has been shown to find optima in 10–50× fewer experiments. Photonic lab automation systems integrate GPU-accelerated spectroscopic analysis (Raman, fluorescence) for real-time compound characterization.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Bayesian optimization with Gaussian process (GP-UCB, qEI), neural network ensemble surrogate, multi-fidelity BO, reinforcement learning (PPO for experiment selection), active learning, parallel batch BO (TurBO), uncertainty quantification via deep ensembles or MC dropout.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/closed-loop-autonomous-self-driving-labs.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/closed-loop-autonomous-self-driving-labs.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\closed-loop-autonomous-self-driving-labs.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: ChEMBL HTS screening data (https://www.ebi.ac.uk/chembl/); Open Reaction Database (ORD) — chemical reaction outcomes (https://open-reaction-database.org/); Therapeutic Data Commons (TDC) — multi-property drug benchmarks (https://tdcommons.ai/); Syngas Fermentation Simulator multi-fidelity dataset (https://arxiv.org/abs/2311.05776).

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

BoTorch (https://github.com/pytorch/botorch) — GPU Bayesian optimization with PyTorch; Ax (https://github.com/facebook/Ax) — adaptive experimentation platform using BoTorch; Summit (https://github.com/sustainable-processes/summit) — BO library for chemical process optimization; Olympus (https://github.com/aspuru-guzik-group/olympus) — benchmark framework for self-driving lab algorithms.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuDNN for deep ensemble surrogate inference, Cholesky factorization via cuSolver for GP posterior, GPU-accelerated acquisition function optimization (batch gradient ascent); pattern: prior experiment observations → GPU GP/neural surrogate fit → parallel acquisition function maximization (256 candidates) → top-k experiments dispatched to robot → new measurements update surrogate. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
