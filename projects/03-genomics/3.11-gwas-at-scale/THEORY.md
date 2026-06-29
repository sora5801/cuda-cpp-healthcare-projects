# THEORY — 3.11 GWAS at Scale

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. Diagrams in Mermaid/ASCII
> are welcome. See [README.md](README.md) for the quick tour and build steps.
>
> _Educational only — not for clinical use._

<!-- =======================================================================
     The block below is the verbatim catalog deep-dive for this project,
     stamped in by scaffold.py as raw material. Use it to write the sections
     that follow, then DELETE it (or fold it into "The science"). Every
     TODO(theory) below must be completed before the project is "done".
     ======================================================================= -->

<details>
<summary>Catalog deep-dive (raw source material — fold into the sections below, then remove)</summary>

### 3.11 GWAS at Scale 🟡 · Active R&D
- **Deep dive:** Genome-wide association studies test millions of genetic variants (SNPs) for association with phenotypes, requiring mixed linear model (LMM) corrections to control population stratification. The computational bottleneck is constructing the genetic relatedness matrix (GRM), an N×N matrix of pairwise genomic similarity across N individuals (N ~ 500 k in UK Biobank), and then fitting LMM scores per variant. On GPU, the GRM is a large dense matrix multiply of the genotype matrix (N × M) with itself, directly accelerated by cuBLAS GEMM. GWAS-Flow (GPU) exploits this for a fast LMM approximation, and RAPIDS GPU-GWAS uses GPU-native logistic regression across all variants simultaneously.
- **Key algorithms:** Linear mixed model (LMM) with LOCO correction; genetic relatedness matrix (GRM) construction via GEMM; ridge regression / BOLT-LMM variance component estimation; logistic regression per variant; principal component analysis (PCA) for stratification correction.
- **Datasets:** UK Biobank — 500 k individuals, 800 k variants (https://www.ukbiobank.ac.uk/); GWAS Catalog — curated published associations (https://www.ebi.ac.uk/gwas/); dbGaP — controlled-access GWAS datasets (https://www.ncbi.nlm.nih.gov/gap/); gnomAD LD reference panels (https://gnomad.broadinstitute.org/).
- **Starter repos/tools:** GWAS-Flow (https://www.biorxiv.org/content/10.1101/783100) — GPU LMM-based GWAS framework; GPU-GWAS / G2WAS (https://github.com/STRIDES-Codes/GPU-GWAS) — RAPIDS-based GPU GWAS pipeline; REGENIE (https://github.com/rgcgithub/regenie) — whole-genome regression GWAS (CPU, GPU integration target); PLINK2 (https://www.cog-genomics.org/plink/2.0/) — CPU reference with GPU matrix paths via OpenBLAS/MKL.
- **CUDA libraries & GPU pattern:** cuBLAS DGEMM for GRM construction (N×M times M×N); cuSolver for matrix decomposition; RAPIDS cuDF for genotype data loading; cuML logistic regression per-SNP in batches; multi-GPU via NCCL for large N.

</details>

---

## 1. The science

TODO(theory): The biology / medicine / physics being modeled — enough for a
reader to understand the *problem* before any math. What real-world question
does computing this answer?

## 2. The math

TODO(theory): The governing equations / formal problem statement, with **every
symbol defined** (units, ranges). State inputs, outputs, and the objective.

## 3. The algorithm

TODO(theory): Step-by-step. Include **complexity analysis**: serial cost vs. the
parallel work/depth. Where is the arithmetic intensity? What is the data-access
pattern?

## 4. The GPU mapping

TODO(theory): How the algorithm becomes **threads / blocks / grids**.
- Thread-to-data mapping (which thread owns which element).
- Launch configuration and the reasoning (block size, grid size).
- Memory hierarchy used and **why**: global / shared / registers / constant /
  texture. Where is the bandwidth bottleneck? What is the occupancy story?
- Which CUDA library (cuBLAS / cuFFT / cuRAND / cuSOLVER / Thrust) does what,
  and what it would take to write that step by hand (no black boxes — §6.1.6).

```
TODO(theory): an ASCII or Mermaid diagram of the grid/block decomposition.
```

## 5. Numerical considerations

TODO(theory): Precision (FP32 vs FP64) and why. Stability. Race conditions and
whether atomics are used. **Determinism**: does the parallel reduction reorder
floating-point sums? If so, say so and quantify the caveat.

## 6. How we verify correctness

TODO(theory): The CPU reference (`src/reference_cpu.cpp`), the **tolerance** and
why that value, and the edge cases checked. Explain why agreement between an
independent serial implementation and the GPU implementation is convincing
evidence of correctness.

## 7. Where this sits in the real world

TODO(theory): How production tools (named in the catalog "Prior art") do this
differently — what they add (scale, accuracy, features) that this teaching
version omits. If this is a 🔴 frontier project shipped as a reduced-scope
teaching version, describe the full approach here.

---

## References

TODO(theory): Papers, docs, and the starter repos from the catalog, with one
line each on what to learn from them.
