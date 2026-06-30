# 3.11 — GWAS at Scale

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.11`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

A **genome-wide association study (GWAS)** asks, for each of millions of genetic
variants (SNPs), whether carrying more copies of one allele shifts a trait (height,
LDL cholesterol, disease risk). This project builds the two computations that
dominate a GWAS and that the GPU accelerates: (1) the **genetic relatedness matrix
(GRM)** — an N×N matrix of how genetically similar every pair of individuals is,
built as a single dense matrix multiply with **cuBLAS DGEMM**; and (2) a **per-SNP
association scan** — a single-marker linear regression for every variant, run as
one GPU thread per SNP. On a clearly-labeled **synthetic** cohort with a handful
of injected *causal* SNPs, a correct pipeline ranks exactly those SNPs at the top —
that recovery is the demo's headline, and the GPU result is checked entry-by-entry
against a plain-C++ reference.

## What this computes & why the GPU helps

Genome-wide association studies test millions of genetic variants (SNPs) for association with phenotypes, requiring mixed linear model (LMM) corrections to control population stratification. The computational bottleneck is constructing the genetic relatedness matrix (GRM), an N×N matrix of pairwise genomic similarity across N individuals (N ~ 500 k in UK Biobank), and then fitting LMM scores per variant. On GPU, the GRM is a large dense matrix multiply of the genotype matrix (N × M) with itself, directly accelerated by cuBLAS GEMM. GWAS-Flow (GPU) exploits this for a fast LMM approximation, and RAPIDS GPU-GWAS uses GPU-native logistic regression across all variants simultaneously.

**The parallel bottleneck:** the GRM is `GRM = (1/M)·Z·Zᵀ`, a dense `N×N` output
formed from `N×M` standardized genotypes — an `O(N²M)` matrix multiply. For
UK-Biobank scale (`N≈500k`, `M≈800k`) this is the single most expensive step, and
it is *exactly* what GPU GEMM was built for: cuBLAS DGEMM saturates the device's
floating-point and memory bandwidth far beyond a CPU triple loop. The per-SNP
regression scan is a second, embarrassingly parallel bottleneck (`M` independent
fits) handled by one thread per SNP.

## The algorithm in brief

Linear mixed model (LMM) with LOCO correction; genetic relatedness matrix (GRM) construction via GEMM; ridge regression / BOLT-LMM variance component estimation; logistic regression per variant; principal component analysis (PCA) for stratification correction.

This teaching version implements the two load-bearing pieces end-to-end:

- **Standardize** each SNP column to a z-score `(g − 2p)/√(2p(1−p))` (the GCTA/VanRaden normalization).
- **GRM** `= (1/M)·Z·Zᵀ` via **cuBLAS DGEMM** (`compute_75…89`, double precision).
- **Per-SNP single-marker regression**: slope `β = Σxy/Σx²`, standard error, `t`,
  `χ² = t²`, and `−log₁₀(p)` — one GPU thread per SNP.
- **Recover** the injected causal SNPs by ranking on `χ²`.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation, including how the LMM and PCA stratification correction extend this.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)). The project links
**cuBLAS** (`cublas.lib`) for the GRM matrix multiply — already wired into the
`.vcxproj` and `CMakeLists.txt`.

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

The demo builds if needed, runs on `data/sample/gwas_sample.txt`, prints the
result, shows the GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/gwas_sample.txt` — a tiny, **synthetic**
  cohort (200 individuals × 60 SNPs, 5 injected causal SNPs) so the demo runs with
  zero downloads. Generated deterministically by `scripts/make_synthetic.py`.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print how to apply for and
  fetch real cohorts (they never bypass access controls).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: UK Biobank — 500 k individuals, 800 k variants (https://www.ukbiobank.ac.uk/); GWAS Catalog — curated published associations (https://www.ebi.ac.uk/gwas/); dbGaP — controlled-access GWAS datasets (https://www.ncbi.nlm.nih.gov/gap/); gnomAD LD reference panels (https://gnomad.broadinstitute.org/).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): the
program reports the GRM diagnostics, that **all 5 injected causal SNPs are
recovered in the top 10**, the ranked top hits, and `RESULT: PASS`. The result is
computed on both the **GPU** (`src/kernels.cu`) and a **CPU reference**
(`src/reference_cpu.cpp`); main.cu asserts they agree:

- **GRM** entrywise within `1e-9` (they actually match to ~`1e-16`; the tolerance
  leaves room for the GPU's different summation order — see THEORY §Numerics).
- **Association** `χ²` per SNP within `1e-9` (matches to ~`1e-13`).

That agreement, plus the recovery of the known causal SNPs, is the correctness
guarantee. Timings are printed to **stderr** (shown but not part of the check).

## Code tour

Read in this order:

1. [`src/gwas_core.h`](src/gwas_core.h) — the shared `__host__ __device__` math
   (standardization + the single-marker regression formula). The CPU and GPU both
   call these, which is *why* their numbers match.
2. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
4. [`src/kernels.cu`](src/kernels.cu) — the standardize kernel, the **cuBLAS DGEMM**
   GRM build (explained, not a black box), and the per-SNP association kernel.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

GWAS-Flow (https://www.biorxiv.org/content/10.1101/783100) — GPU LMM-based GWAS framework; GPU-GWAS / G2WAS (https://github.com/STRIDES-Codes/GPU-GWAS) — RAPIDS-based GPU GWAS pipeline; REGENIE (https://github.com/rgcgithub/regenie) — whole-genome regression GWAS (CPU, GPU integration target); PLINK2 (https://www.cog-genomics.org/plink/2.0/) — CPU reference with GPU matrix paths via OpenBLAS/MKL.

What to learn from each:

- **GWAS-Flow** — how the GRM (the DGEMM we implement here) feeds a mixed-linear-model
  variance-component fit that controls for relatedness.
- **GPU-GWAS / G2WAS** — a RAPIDS pipeline that runs per-SNP regression on the GPU,
  the same parallelization as our `assoc_kernel`, at biobank scale.
- **REGENIE** — the modern two-step whole-genome-regression GWAS; its step 1 ridge
  regression is the natural next extension (see Exercises).
- **PLINK2** — the field-standard CPU tool; its `.bed`/`.pgen` formats are what a
  production loader would parse instead of our text sample.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuBLAS DGEMM for GRM construction (N×M times M×N); cuSolver for matrix decomposition; RAPIDS cuDF for genotype data loading; cuML logistic regression per-SNP in batches; multi-GPU via NCCL for large N.

Concretely (mapping to PATTERNS.md): the GRM is the **"dense linear algebra → use
cuBLAS"** pattern (like flagship `2.06`), and the per-SNP scan is the **"independent
jobs"** pattern (one thread per SNP, like flagship `1.12` Tanimoto). The shared
`__host__ __device__` core (PATTERNS.md §2) makes CPU/GPU verification near-exact.

## Exercises

1. **PCA stratification correction.** Compute the top eigenvectors of the GRM with
   **cuSOLVER `Dsyevd`** (see flagship `2.06`) and add them as covariates to the
   per-SNP regression. Watch spurious associations from population structure shrink.
2. **Logistic regression.** Swap the linear single-marker test for a logistic one
   (binary case/control phenotype) via Newton–Raphson per SNP — still one thread per
   SNP, now with an inner IRLS loop.
3. **Genomic control λ.** Compute the median `χ²` across SNPs divided by `0.4549`
   (the median of `χ²₁`); a value ≫ 1 signals inflation. Print it and tune the noise
   in `make_synthetic.py` to see it move.
4. **Tile the GRM at scale.** For `N` too large for one DGEMM, block the multiply
   into tiles and accumulate — the strategy GWAS-Flow uses for biobank `N`.
5. **Float vs double.** Re-run the GRM in single precision (`cublasSgemm`) and
   measure how the verification error grows; decide whether FP32 is acceptable.

## Limitations & honesty

- **Synthetic data, labeled as such.** The committed cohort is generated by
  `scripts/make_synthetic.py` with a fixed seed; it is *not* real patient data, and
  the recovered associations are an artifact of the injected effects. No clinical or
  biological claim is implied.
- **Reduced scope.** A production GWAS uses a **mixed linear model** (LMM) to
  correct for relatedness/structure (BOLT-LMM, REGENIE); this teaching version
  builds the GRM that an LMM *needs* and runs the *naive* single-marker scan on top
  of it, but does **not** fit the variance components or apply LOCO. THEORY.md
  §"real world" spells out the gap.
- **No multiple-testing or LD handling.** We print raw `−log₁₀(p)`; a real study
  applies a genome-wide threshold (`5×10⁻⁸`) and accounts for linkage disequilibrium.
- **`t ≈ Normal`.** For large `N` we approximate the Student-t tail by a standard
  normal; for tiny `N` the exact `t_{N−2}` tail would differ slightly.
- **Tiny inputs are launch/copy-bound.** The printed GPU timings are a *teaching
  artifact*, not a benchmark — the DGEMM's advantage only appears as `N` and `M` grow.
