# 13.12 — Exposure-Response & Dose-Response Modelling

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Pharmacology%20%26%20Clinical%20Quantitative%20Modeling-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 13: Pharmacology & Clinical Quantitative Modeling · Catalog ID `13.12`
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

Quantifies the relationship between drug exposure metrics (AUC, Cmax, trough) and clinical or safety endpoints (tumour response, biomarker change, toxicity probability) using GPU-accelerated nonlinear regression and machine learning. In dose-finding trials (Phase I/II), Bayesian model-based dose-escalation designs (EWOC, mTPI-2, BLRM) require rapid posterior sampling after each dose cohort — GPU-accelerated MCMC provides the turnaround speed needed for within-day decision support. Sigmoidal Emax models, logistic regression, and exposure-toxicity models are fitted to cumulative clinical datasets with GPU-parallel gradient computation. The key bottleneck is running thousands of simulated future trial realisations in parallel for adaptive design decision criteria.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Sigmoidal Emax / Hill equation fitting, Bayesian Logistic Regression Model (BLRM), Escalation With Overdose Control (EWOC), modified Toxicity Probability Interval (mTPI), Emax-time models, power models, direct vs. indirect response PD models, mixture models for responder/non-responder subpopulations, concordance dose-response index.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/exposure-response-dose-response-modelling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/exposure-response-dose-response-modelling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\exposure-response-dose-response-modelling.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: FDA Pharmacometrics Reviews — dose-response data from NDA/BLA submissions (https://www.fda.gov/drugs/drug-approvals-and-databases/pharmacometrics-reviews) Published dose-escalation trial data in Oncology (verify individual publications) DoseFinding R package example datasets (verify URL) CDISC ADaM dose-response trial data formats (https://www.cdisc.org/)

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

Pumas (https://pumas.ai/) — GPU Bayesian E-R modelling in Julia DoseFinding R package (https://cran.r-project.org/web/packages/DoseFinding/) — classical dose-finding model fitting BOIN (https://cran.r-project.org/web/packages/BOIN/) — Bayesian optimal interval design for dose-finding trialDesign (verify URL) — simulation platform for adaptive dose-escalation

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuRAND for Monte Carlo posterior simulation, cuBLAS for sigmoidal Emax regression Hessians, custom CUDA kernels for parallel trial simulation over candidate dose levels; pattern: GPU-parallel simulation of thousands of adaptive trial scenarios for decision support. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
