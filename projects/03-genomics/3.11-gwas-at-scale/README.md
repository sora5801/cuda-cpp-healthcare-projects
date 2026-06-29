# 3.11 — GWAS at Scale

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.11`
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

Genome-wide association studies test millions of genetic variants (SNPs) for association with phenotypes, requiring mixed linear model (LMM) corrections to control population stratification. The computational bottleneck is constructing the genetic relatedness matrix (GRM), an N×N matrix of pairwise genomic similarity across N individuals (N ~ 500 k in UK Biobank), and then fitting LMM scores per variant. On GPU, the GRM is a large dense matrix multiply of the genotype matrix (N × M) with itself, directly accelerated by cuBLAS GEMM. GWAS-Flow (GPU) exploits this for a fast LMM approximation, and RAPIDS GPU-GWAS uses GPU-native logistic regression across all variants simultaneously.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Linear mixed model (LMM) with LOCO correction; genetic relatedness matrix (GRM) construction via GEMM; ridge regression / BOLT-LMM variance component estimation; logistic regression per variant; principal component analysis (PCA) for stratification correction.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/gwas-at-scale.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/gwas-at-scale.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\gwas-at-scale.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: UK Biobank — 500 k individuals, 800 k variants (https://www.ukbiobank.ac.uk/); GWAS Catalog — curated published associations (https://www.ebi.ac.uk/gwas/); dbGaP — controlled-access GWAS datasets (https://www.ncbi.nlm.nih.gov/gap/); gnomAD LD reference panels (https://gnomad.broadinstitute.org/).

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

GWAS-Flow (https://www.biorxiv.org/content/10.1101/783100) — GPU LMM-based GWAS framework; GPU-GWAS / G2WAS (https://github.com/STRIDES-Codes/GPU-GWAS) — RAPIDS-based GPU GWAS pipeline; REGENIE (https://github.com/rgcgithub/regenie) — whole-genome regression GWAS (CPU, GPU integration target); PLINK2 (https://www.cog-genomics.org/plink/2.0/) — CPU reference with GPU matrix paths via OpenBLAS/MKL.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuBLAS DGEMM for GRM construction (N×M times M×N); cuSolver for matrix decomposition; RAPIDS cuDF for genotype data loading; cuML logistic regression per-SNP in batches; multi-GPU via NCCL for large N. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
