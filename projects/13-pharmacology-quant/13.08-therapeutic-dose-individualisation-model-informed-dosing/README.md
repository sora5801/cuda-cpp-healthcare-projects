# 13.8 — Therapeutic Dose Individualisation / Model-Informed Dosing

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Pharmacology%20%26%20Clinical%20Quantitative%20Modeling-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 13: Pharmacology & Clinical Quantitative Modeling · Catalog ID `13.8`
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

Adapts drug dosing for individual patients using Bayesian updating of a population PK/PD prior with the patient's own concentration measurements (therapeutic drug monitoring, TDM). GPU acceleration is relevant in three ways: (1) population model fitting on GPU (as in 13.1); (2) real-time posterior ODE integration for thousands of candidate dose levels simultaneously to find the optimal dose; (3) simulation-based model averaging across uncertainty in individual parameters. The AUC-target dosing problem reduces to: for each candidate dose schedule, integrate the PK ODE forward for 30 days and check whether AUC hits target — parallelised across doses on GPU. Pumas and Bayesian NONMEM implement this on GPU.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Bayesian individual parameter estimation (MAP, full Bayes), AUC-target optimisation via GPU-parallel ODE forward simulation, MAP-adaptive dosing, Model Predictive Control (MPC) for infusion rate optimisation, optimal sampling time selection (D-optimal), individual dose prediction with uncertainty propagation, neural ODE for personalised PK.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/therapeutic-dose-individualisation-model-informed-dosing.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/therapeutic-dose-individualisation-model-informed-dosing.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\therapeutic-dose-individualisation-model-informed-dosing.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Published TDM datasets (vancomycin, aminoglycosides, tacrolimus) — available through PharmPK listserv (verify URL) NONMEM example datasets — shipped with NONMEM installation (verify URL) Latent Neural-ODE paper dataset (https://arxiv.org/abs/2602.03215) — personalised dosing with neural ODE MIMIC-IV medication and lab data — vancomycin AUC retrospective cohorts (https://physionet.org/content/mimiciv/)

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

Pumas (https://pumas.ai/) — GPU-accelerated Bayesian dose individualisation in Julia Torsten (https://github.com/metrumresearchgroup/Torsten) — Stan extension for PK ODE solving; Bayesian TDM InsightRx (verify URL) — commercial Bayesian dosing platform BayesPK (verify URL) — open-source Bayesian PK software for TDM

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom CUDA ODE kernels for forward simulation across dose grid, cuRAND for uncertainty sampling, Thrust for AUC computation; pattern: dose-grid-parallel ODE integration — each CUDA thread evaluates one dose schedule forward simulation. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
