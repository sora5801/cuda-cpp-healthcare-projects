# 13.18 — Pharmacogenomics-Guided Precision Dosing

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Pharmacology%20%26%20Clinical%20Quantitative%20Modeling-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 13: Pharmacology & Clinical Quantitative Modeling · Catalog ID `13.18`
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

Integrates individual genetic variants (CYP2D6, CYP2C19, VKORC1, SLCO1B1, UGT1A1) with demographic covariates and drug-specific models to predict optimal starting doses for precision medicine. GPU parallelism enables simulation of the full population variant space: with 50 common pharmacogenomic variants each having 2–3 allele combinations, the state space has ~10¹⁰ genotype combinations, requiring Monte Carlo sampling across thousands of virtual genotype profiles simultaneously. GWAS-based PK prediction models (GNN + genomic embedding) trained on biobank-scale data (UK Biobank + All of Us) are GPU-training-bound. Deep learning ensemble models that integrate genotype × drug × phenotype interactions require massive batched forward passes on GPU.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

CPIC (Clinical Pharmacogenomics Implementation Consortium) rules, population PK covariate models incorporating genotype, GWAS-PK association studies, GNN on drug metabolic pathway graphs, random forest / gradient boosting with genomic features, Bayesian network for genotype-phenotype-PK integration, polygenic PK scores.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/pharmacogenomics-guided-precision-dosing.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/pharmacogenomics-guided-precision-dosing.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\pharmacogenomics-guided-precision-dosing.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: PharmGKB — curated gene-drug relationships with evidence levels (https://www.pharmgkb.org/) CPIC guidelines data — actionable pharmacogenomic recommendations (https://cpicpgx.org/) UK Biobank genotype + prescription data — 500k individuals (https://www.ukbiobank.ac.uk/) All of Us pharmacogenomics cohort (https://allofus.nih.gov/)

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

PyPGx (https://github.com/sbslee/pypgx) — pharmacogenomics genotyping from next-generation sequencing PharmCAT (https://github.com/PharmGKB/PharmCAT) — pharmacogenomics clinical annotation tool Pumas PGx module (https://pumas.ai/) — pharmacogenomics-integrated PK/PD in Julia (verify URL) SAIGE-GPU (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12960912/) — GPU GWAS for PK covariate discovery (verify GitHub URL)

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuSPARSE for genotype matrix operations, cuDNN for neural PGx-PK models, cuRAND for Monte Carlo genotype space exploration; pattern: GPU-parallel simulation across thousands of genotype profiles × dose schedules. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
