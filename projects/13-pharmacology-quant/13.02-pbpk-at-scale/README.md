# 13.2 — PBPK at Scale

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Pharmacology%20%26%20Clinical%20Quantitative%20Modeling-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 13: Pharmacology & Clinical Quantitative Modeling · Catalog ID `13.2`
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

Physiologically based pharmacokinetic (PBPK) models describe drug disposition through ~15 interconnected physiological compartments (blood, liver, kidney, lung, fat, muscle, etc.), each defined by ODEs parameterised by tissue volumes, blood flows, and metabolic rate constants. High-throughput virtual screening of thousands of compounds requires solving the full PBPK ODE system (30–60 ODEs) for each compound simultaneously — a batch of 10,000 compounds is 600,000 simultaneous ODEs, well-suited to GPU-parallel Runge-Kutta integration. NVIDIA's nvQSP implements a GPU-accelerated RODAS4 stiff ODE solver specifically for QSP/PBPK population studies. Monte Carlo virtual population simulations (500–5000 virtual subjects per compound) further multiply the parallelism requirement.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

RODAS4 stiff ODE solver (GPU implementation), Runge-Kutta 4/5, adaptive stepsize control, PBPK parameter estimation via Bayesian MCMC, machine-learning-predicted ADME inputs (logP, Vd, CLint), tissue-plasma partition coefficient estimation (Rodgers-Rowland, Berezhkovskiy).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/pbpk-at-scale.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/pbpk-at-scale.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\pbpk-at-scale.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Open Systems Pharmacology PBPK model repository (https://github.com/Open-Systems-Pharmacology/OSP-PBPK-Model-Library) — 100+ validated human PBPK models DrugBank ADME data — 14k+ drugs with physicochemical and metabolic parameters (https://www.drugbank.com/) FDA/EMA drug approval submission PK data — publicly available pharmacokinetic data from drug labels (verify URL) ChEMBL ADMET data — assay-based ADME measurements (https://www.ebi.ac.uk/chembl/)

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

PK-Sim (https://github.com/Open-Systems-Pharmacology/PK-Sim) — open-source whole-body PBPK software (C#; GPU via OSP Suite) nvQSP (https://github.com/NVIDIA-Digital-Bio/nvQSP) — NVIDIA GPU-accelerated QSP/PBPK ODE solvers (CUDA) SimBiology (MATLAB) — PBPK modelling with parallel computing toolbox for GPU (verify URL) PBPKsim (verify URL) — Python PBPK simulation framework

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom CUDA RODAS4/RK45 stiff ODE solver kernels, cuBLAS for Jacobian evaluation, Thrust for adaptive stepsize selection; pattern: one CUDA thread block per virtual subject, with ODE compartments mapped to shared memory. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
