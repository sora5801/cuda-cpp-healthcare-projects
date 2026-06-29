# 13.6 — Quantitative Systems Pharmacology

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Pharmacology%20%26%20Clinical%20Quantitative%20Modeling-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 13: Pharmacology & Clinical Quantitative Modeling · Catalog ID `13.6`
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

QSP models integrate pharmacokinetics with mechanistic biology (immune signalling, tumour growth, disease pathway models) through large ODE systems (100–10,000 equations). Stiff ODE integration dominates compute: a QSP model with 1,000 equations × 1,000 virtual patients requires solving 10⁶ coupled ODEs simultaneously. NVIDIA's nvQSP implements GPU-accelerated RODAS4 (an L-stable solver for stiff systems) specifically for this purpose, achieving orders-of-magnitude speedup. Virtual twin patient simulations for oncology trials run thousands of patient ODEs simultaneously, with GPU thread blocks each solving one patient's equation system. Physics-Informed Neural Networks (PINNs) are emerging as GPU-native surrogates that learn QSP system dynamics from data.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

RODAS4/LSODA stiff ODE integration, sensitivity analysis (forward/adjoint), global parameter search (population Monte Carlo, ABC), PBPK-QSP coupling, immune checkpoint model ODEs (anti-PD1, CAR-T dynamics), tumour growth inhibition models, Physics-Informed Neural Networks (PINNs), QSP model reduction (MBAM).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/quantitative-systems-pharmacology.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/quantitative-systems-pharmacology.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\quantitative-systems-pharmacology.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: QSP model repository (DDMoRe consortium) — interoperable QSP models (https://www.ddmore.eu/) BioModels Database — 2000+ curated mathematical models of biological processes (https://www.ebi.ac.uk/biomodels/) NIH Systems Biology Data (verify URL) — mechanistic pathway data Open Systems Pharmacology QSP library (https://github.com/Open-Systems-Pharmacology/QSP-PK-Model-Library)

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

nvQSP (https://github.com/NVIDIA-Digital-Bio/nvQSP) — NVIDIA GPU-accelerated QSP ODE solver (CUDA RODAS4) SBML/Tellurium (https://github.com/sys-bio/tellurium) — systems biology model simulator; GPU backend emerging SBMLtoODEjl (verify URL) — Julia ODE generator from SBML for GPU integration via CUDA.jl Copasi (https://copasi.org/) — biochemical network simulator; parallel via COPASI MPI interface

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom CUDA RODAS4 stiff ODE kernels, cuBLAS for Jacobian LU factorisation, cuSPARSE for sparse ODE right-hand-side; pattern: one CUDA thread block per virtual patient, each thread within block updates one ODE compartment per step. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
