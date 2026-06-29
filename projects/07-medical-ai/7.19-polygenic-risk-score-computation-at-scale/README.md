# 7.19 — Polygenic Risk Score Computation at Scale

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20AI%20%26%20Clinical%20Deep%20Learning-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 7: Medical AI & Clinical Deep Learning · Catalog ID `7.19`
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

Computes polygenic risk scores (PRS) for millions of individuals by summing effect sizes from thousands to millions of GWAS-identified SNPs across the genome. The core operation is a large sparse matrix-vector multiply: individual genotype matrix (N_samples × M_SNPs, typically stored as INT2 or INT8 allele dosages) times a weight vector of SNP effect sizes. For UK Biobank scale (500k individuals × 6M SNPs), this is a 3 TB sparse matrix multiply best suited to GPU execution via cuSPARSE. SAIGE-GPU accelerates mixed-model GWAS (which underlies PRS weight estimation) using GPU-optimised linear algebra, enabling phenome-wide PRS across hundreds of traits simultaneously.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Clumping and Thresholding (C+T), LDpred2, PRS-CS, lassosum, SAIGE mixed-model GWAS on GPU, LD pruning, population stratification correction (PCA), multi-ancestry PRS meta-analysis.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/polygenic-risk-score-computation-at-scale.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/polygenic-risk-score-computation-at-scale.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\polygenic-risk-score-computation-at-scale.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: UK Biobank — 500k WGS individuals, 7000 phenotypes (https://www.ukbiobank.ac.uk/) All of Us Research Program — >680k diverse participants, whole genome sequencing (https://allofus.nih.gov/) FinnGen — 500k Finnish participants with national registry linkage (https://www.finngen.fi/en) dbGaP GWAS Summary Stats — thousands of published GWAS across traits (https://www.ncbi.nlm.nih.gov/gap/)

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

SAIGE-GPU (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12960912/) — GPU-accelerated mixed-model GWAS (verify GitHub URL) PRSice-2 (https://github.com/choishingwan/PRSice) — PRS computation tool (CPU; GPU via matrix backend) LDpred2 (https://github.com/privefl/bigsnpr) — Bayesian PRS in R with parallelism via bigstatsr PLINK 2.0 (https://www.cog-genomics.org/plink/2.0/) — genome-wide association toolkit with GPU-accelerated linear algebra

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuSPARSE SpMV for genotype × effect-size matrix, cuBLAS for LD matrix operations, Thrust for sorting variant effect sizes; pattern: chunked sparse matrix multiply with INT2 genotype encoding to maximise VRAM utilisation. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
