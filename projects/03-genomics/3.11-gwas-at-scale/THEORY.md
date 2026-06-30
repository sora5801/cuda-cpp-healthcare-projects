# THEORY — 3.11 GWAS at Scale

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

Every human genome differs from the reference at millions of single positions
called **SNPs** (single-nucleotide polymorphisms). At a biallelic SNP an
individual carries 0, 1, or 2 copies of the *minor* (less common) allele — this
count is the **dosage**, and under the *additive model* it is the genotype we
analyze.

A **genome-wide association study (GWAS)** takes a cohort of `N` people, each
measured for a **phenotype** `y` (a quantitative trait like height or LDL
cholesterol, or a binary disease status) and genotyped at `M` SNPs, and asks for
**each SNP independently**: *does carrying more of this allele shift the trait?*
A SNP whose dosage correlates with `y` beyond chance is "associated" — a clue
that this region of the genome influences the trait.

Two facts make this hard at scale and motivate the GPU:

1. **Relatedness / population structure.** People are not independent: relatives
   and members of the same ancestry group share alleles *and* environments. If
   ignored, that shared structure creates **spurious associations** (a SNP that
   merely tags ancestry looks "associated" with any ancestry-correlated trait).
   The standard fix is to model relatedness explicitly via the **genetic
   relatedness matrix (GRM)** — an `N×N` matrix where entry `(a,b)` measures how
   genetically similar individuals `a` and `b` are. The GRM is the kernel of a
   **mixed linear model (LMM)** that corrects the test for structure.

2. **Volume.** UK Biobank has `N≈500,000` individuals and `M≈800,000` SNPs.
   Building the GRM is an `O(N²M)` dense matrix multiply; scanning `M` SNPs is a
   second large, parallel workload. Both are textbook GPU jobs.

This project implements the two GPU-accelerated cores — **GRM construction** and
the **per-SNP association scan** — on a clearly-synthetic cohort, and verifies
them against a CPU reference.

## 2. The math

**Inputs.** A genotype matrix `G ∈ {0,1,2}^{N×M}` (row = individual `i`, column =
SNP `j`, entry `g_ij` = minor-allele dosage), and a phenotype vector
`y ∈ ℝ^N`.

**Standardization (the Z matrix).** For SNP `j` let `p_j` be its minor-allele
frequency, estimated as `p_j = (Σ_i g_ij) / (2N)` (each of the `N` people carries
up to 2 alleles). Under Hardy–Weinberg equilibrium a dosage has mean `2p_j` and
variance `2p_j(1−p_j)`. We standardize each entry to a **z-score**

```
z_ij = (g_ij − 2 p_j) / sqrt( 2 p_j (1 − p_j) ).
```

This is the GCTA/VanRaden normalization: centering removes the SNP's baseline,
and scaling by the Hardy–Weinberg standard deviation up-weights rare-allele
sharing (which is more informative about relatedness). Call the result
`Z ∈ ℝ^{N×M}`.

**Genetic relatedness matrix (GRM).** The relatedness of individuals `a` and `b`
is their standardized genotypes' dot product, averaged over SNPs:

```
GRM[a][b] = (1/M) Σ_j z_aj z_bj          ⇔        GRM = (1/M) · Z · Zᵀ.
```

`GRM` is symmetric `N×N`. The diagonal `GRM[a][a] ≈ 1` (self-relatedness);
off-diagonals are `≈ 0` for unrelated people and larger for relatives or
same-ancestry pairs.

**Single-marker association (the per-SNP test).** Mean-center `y` (so the
intercept drops out: `ȳ = 0`). For SNP `j` fit the ordinary-least-squares line
`y_i = β_j z_ij + e_i`. Because both `z·j` and `y` are centered, the slope is the
covariance/variance ratio

```
β_j = (Σ_i z_ij y_i) / (Σ_i z_ij²) = s_xy / s_xx.
```

With residual sum of squares `SSE = s_yy − β_j s_xy` and `σ̂² = SSE/(N−2)`, the
slope's standard error, test statistic, and p-value are

```
SE(β_j) = sqrt( σ̂² / s_xx ),   t_j = β_j / SE(β_j),
χ²_j = t_j²  (~ χ²₁ under H₀),   p_j = 2·P(Z > |t_j|),   score_j = −log₁₀ p_j.
```

The three **sufficient statistics** `s_xx = Σz², s_xy = Σzy, s_yy = Σy²` are all
a SNP's column needs — this is what each GPU thread accumulates.

## 3. The algorithm

```
1. load G (N×M int8) and y (N)                              O(NM) I/O
2. per-SNP p_j, sd_j   (one column sum each)                O(NM)
3. standardize: z_ij = (g_ij − 2p_j)/sd_j                   O(NM)
4. GRM = (1/M) Z Zᵀ                                         O(N²M)   <-- dominant
5. center y; per SNP accumulate (s_xx,s_xy,s_yy) -> β,t,χ²  O(NM)
6. rank SNPs by χ²; report top hits                         O(M log M)
```

**Complexity.** Step 4 (the GRM) is the giant: `O(N²M)` multiply–adds producing
`N²` outputs — for biobank sizes, petaFLOP-scale. Step 5 (the scan) is `O(NM)`
but with `M` *independent* fits. The serial CPU cost of step 4 is a triple loop;
the GPU collapses it into one cuBLAS call.

**Arithmetic intensity.** The GRM multiply has high arithmetic intensity
(`O(M)` FLOPs per output reused across the row/column), which is why a tuned GEMM
approaches peak FLOPs. The standardize and scan kernels are **bandwidth-bound**
(a few FLOPs per byte read), so their cost tracks memory throughput.

## 4. The GPU mapping

Two independent kernels plus one library call:

**(a) `standardize_kernel` — element-wise, 2-D grid.**
Each thread owns one matrix entry `Z[row][col]` and applies the shared
`gwas::standardize`. Block `16×16 = 256` threads; grid covers `⌈M/16⌉ × ⌈N/16⌉`.
Pure global-memory streaming → bandwidth-bound; no shared memory needed.

**(b) GRM via cuBLAS DGEMM — the library, not a black box.**
We want `GRM[N×N] = (1/M)·Z[N×M]·Zᵀ[M×N]` in **double precision**. cuBLAS is
**column-major**; our `Z` is **row-major**. The identity we exploit: *a row-major
`N×M` buffer is bit-identical to a column-major `M×N` buffer.* Call that view
`Zc` (M rows, N cols). Then in cuBLAS's world the product we want is
`GRMc = Zcᵀ·Zc`, i.e. `cublasDgemm(handle, OP_T, OP_N, N, N, M, &alpha, Zc, M,
Zc, M, &beta, GRMc, N)` with `alpha = 1/M`, `beta = 0`. Because the GRM is
symmetric, its column-major result copies straight back into our row-major
buffer. *What hand-rolling costs:* a competitive GEMM needs shared-memory tiling,
register blocking, double-buffered loads, and bank-conflict-free access patterns
— hundreds of lines that cuBLAS already perfects, so we use it and keep our
attention on the genetics (CLAUDE.md §6.1.6).

**(c) `assoc_kernel` — one thread per SNP (independent-jobs pattern).**
Thread `j` owns SNP `j`: it walks the `N` individuals, standardizes that SNP's
dosage on the fly (so we never store a second copy of `Z`), accumulates
`(s_xx, s_xy, s_yy)` in registers, and calls the shared
`gwas::assoc_from_sufficient_stats`. Block 256 threads; grid `⌈M/256⌉`. The
phenotype `y` is re-read by every thread (a natural caching candidate, left
simple for teaching). No atomics: each thread writes its own `AssocResult`.

```
Genotype matrix G  (N individuals × M SNPs)
                 SNP j
              ┌───────────────┐
   indiv i →  │ . . g_ij . . .│   standardize_kernel: thread (col=j,row=i) → Z[i][j]
              │ . . . . . . . │
              └───────────────┘
                     │  Z (N×M)
       cuBLAS DGEMM  ▼  Zᵀ·Z form
              ┌───────────────┐                 assoc_kernel:
   GRM (N×N)  │ 1  r  r  …     │   one thread ── thread j walks column j of G,
              │ r  1  r  …     │   per SNP       accumulates Σz², Σzy, Σy² → χ²_j
              └───────────────┘
```

## 5. Numerical considerations

- **Precision: FP64 throughout.** Relatedness coefficients and `χ²` need the
  dynamic range; we use `double` and `cublasDgemm`. (Exercise 5 explores FP32.)
- **Determinism & summation order.** Floating-point addition is *not* associative.
  The CPU reference sums each GRM dot product left-to-right; cuBLAS DGEMM sums in
  a tiled, parallel order and uses fused multiply-add. So the two GRMs are **not
  guaranteed bit-identical** — on this sample they agree to `~1e-16`, but in
  general the difference can reach `~1e-12…1e-10` for larger `M`. We therefore
  verify to `1e-9`, far below any genetic signal, and *say so* rather than
  pretending exactness (PATTERNS.md §4).
- **No atomics.** Both the scan (one `AssocResult` per thread) and the GRM (cuBLAS
  owns its own reduction) avoid `atomicAdd`, so the GPU output is reproducible
  run-to-run; the program's **stdout is byte-stable** (timings go to stderr).
- **Guards.** Monomorphic SNPs (`p=0` or `1`) would divide by zero in the HWE
  scale; `gwas::hwe_sd` clamps the variance, and `assoc_from_sufficient_stats`
  returns a null result for zero-variance columns. P-values are floored at
  `1e-300` before `log10` so the score stays finite.
- **`t ≈ Normal`.** We approximate the Student-`t_{N−2}` tail by a standard normal
  (`erfc`), exact in the large-`N` limit; both sides use the same `erfc` so they
  agree to the last bit.

## 6. How we verify correctness

`src/reference_cpu.cpp` recomputes **both** outputs with plain serial loops:
`grm_reference` (a triple loop for `(1/M)ZZᵀ`) and `assoc_reference` (per-SNP
sufficient statistics). Crucially, both the reference and the kernels call the
**same** per-element formulas in `gwas_core.h` (the `__host__ __device__` core,
PATTERNS.md §2), so any disagreement can only come from *order of operations*, not
from a different formula.

`main.cu` checks two numbers:

- **GRM**: worst entrywise `|GRM_gpu − GRM_cpu|` over all `N²` entries ≤ `GRM_TOL
  = 1e-9` (observed `~6.7e-16`).
- **Association**: worst `|χ²_gpu − χ²_cpu|` over all `M` SNPs ≤ `ASSOC_TOL =
  1e-9` (observed `~2e-13`).

A second, *scientific* check (not just CPU==GPU): the synthetic data injects 5
**causal** SNPs with known effects, and the demo confirms all 5 are **recovered in
the top 10** by `χ²` rank. Agreement between an independent serial implementation
and the parallel one, *plus* recovery of a planted ground truth, is convincing
evidence the pipeline is correct.

## 7. Where this sits in the real world

This is a deliberately **reduced-scope teaching version**. Production GWAS differs
in several ways the catalog names:

- **Mixed linear model, not naive scan.** Real tools (**BOLT-LMM**, **GWAS-Flow**,
  **REGENIE**) plug the GRM into an LMM: they estimate variance components
  (heritability) and test each SNP *conditional* on the random relatedness effect,
  often with **LOCO** (leave-one-chromosome-out) to avoid proximal contamination.
  We build the GRM that an LMM needs and stop at the single-marker scan.
- **PCA stratification.** Top eigenvectors of the GRM (via **cuSOLVER**) are added
  as covariates to absorb ancestry — Exercise 1, mirroring flagship `2.06`.
- **Logistic regression** for binary phenotypes (case/control), per SNP — the
  `cuML`/`GPU-GWAS` path.
- **Scale & I/O.** `N≈500k` makes the GRM (`N²` doubles ≈ 2 TB) impossible to hold
  whole; production tiles the DGEMM and streams genotypes from **PLINK2**
  `.bed`/`.pgen` or BGEN, loaded with **RAPIDS cuDF**, sharded across GPUs with
  **NCCL**. Our text loader and single-DGEMM are the didactic skeleton of that.
- **Multiple testing & LD.** Real studies apply the genome-wide threshold
  `5×10⁻⁸` and account for linkage disequilibrium; we print raw `−log₁₀p`.

---

## References

- **GCTA / VanRaden GRM** — the standardized `(1/M)ZZᵀ` relatedness estimator we
  build (Yang et al. 2011, *Am J Hum Genet*; VanRaden 2008, *J Dairy Sci*).
- **GWAS-Flow** — https://www.biorxiv.org/content/10.1101/783100 — GPU LMM GWAS;
  shows the GRM→LMM pipeline our GRM feeds.
- **GPU-GWAS / G2WAS** — https://github.com/STRIDES-Codes/GPU-GWAS — RAPIDS
  per-SNP regression on GPU, the same parallelization as `assoc_kernel`.
- **REGENIE** — https://github.com/rgcgithub/regenie — two-step whole-genome
  regression; step-1 ridge is a natural extension (Exercise).
- **PLINK2** — https://www.cog-genomics.org/plink/2.0/ — the CPU field standard and
  the source of the real genotype file formats.
- **cuBLAS DGEMM docs** — the column-major layout convention used in §4(b).
