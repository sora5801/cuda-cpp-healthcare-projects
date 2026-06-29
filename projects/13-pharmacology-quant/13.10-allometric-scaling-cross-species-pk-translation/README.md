# 13.10 — Allometric Scaling & Cross-Species PK Translation

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Pharmacology%20%26%20Clinical%20Quantitative%20Modeling-lightgrey)

> **🟢 Beginner · Established** — Domain 13: Pharmacology & Clinical Quantitative Modeling · Catalog ID `13.10`
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

Translates preclinical animal PK parameters to human predictions using allometric power laws, species-specific physiological scaling, and mechanistic PBPK bridging. When applied at scale — scoring thousands of drug candidates from an in vivo animal study to prioritise compounds for human trials — the PBPK-based cross-species translation requires solving complete animal and human PBPK ODE systems for each candidate. GPU batch ODE integration across thousands of candidates simultaneously is the core acceleration; each candidate requires solving ~15-compartment human and rat/mouse PBPK systems in parallel. Machine learning models (trained on ChEMBL animal-to-human PK datasets) that predict human CL, Vd, and t½ from molecular features are GPU-accelerated via neural forward passes.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Simple allometry (body weight power law), Maximum Lifespan Potential (MLP) correction, Rule of Exponents, PBPK-based cross-species translation, in vitro-in vivo extrapolation (IVIVE), machine-learning regression from molecular descriptors to PK parameters, QSAR-PK modelling.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/allometric-scaling-cross-species-pk-translation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/allometric-scaling-cross-species-pk-translation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\allometric-scaling-cross-species-pk-translation.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: ChEMBL PK dataset — 18k+ compounds with preclinical and human PK data (https://www.ebi.ac.uk/chembl/) Lombardo et al. drug PK dataset — 1352 drugs with CL, Vd, t½ in humans and animals (verify URL) Obach et al. clearance dataset — metabolic clearance measurements (verify URL) Open Systems Pharmacology species parameter databases (https://github.com/Open-Systems-Pharmacology/PK-Sim)

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

PK-Sim (https://github.com/Open-Systems-Pharmacology/PK-Sim) — PBPK allometric scaling built-in pkNCA (https://github.com/billdenney/pknca) — non-compartmental PK analysis in R DeepPK (verify URL) — deep learning PK prediction for allometric scaling ADMET-AI (https://github.com/swansonk14/admet_ai) — ML-based ADME/PK prediction pipeline

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CUDA batched RK4 for species ODE systems, cuBLAS for regression model forward pass, cuDNN for molecular graph encoder; pattern: batch-parallel PBPK translation — one compound per CUDA thread group. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
