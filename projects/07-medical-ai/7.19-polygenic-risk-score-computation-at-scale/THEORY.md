# THEORY — 7.19 Polygenic Risk Score Computation at Scale

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

### 7.19 Polygenic Risk Score Computation at Scale 🟡 · Active R&D

- **Deep dive:** Computes polygenic risk scores (PRS) for millions of individuals by summing effect sizes from thousands to millions of GWAS-identified SNPs across the genome. The core operation is a large sparse matrix-vector multiply: individual genotype matrix (N_samples × M_SNPs, typically stored as INT2 or INT8 allele dosages) times a weight vector of SNP effect sizes. For UK Biobank scale (500k individuals × 6M SNPs), this is a 3 TB sparse matrix multiply best suited to GPU execution via cuSPARSE. SAIGE-GPU accelerates mixed-model GWAS (which underlies PRS weight estimation) using GPU-optimised linear algebra, enabling phenome-wide PRS across hundreds of traits simultaneously.
- **Key algorithms:** Clumping and Thresholding (C+T), LDpred2, PRS-CS, lassosum, SAIGE mixed-model GWAS on GPU, LD pruning, population stratification correction (PCA), multi-ancestry PRS meta-analysis.
- **Datasets:**
  - UK Biobank — 500k WGS individuals, 7000 phenotypes (https://www.ukbiobank.ac.uk/)
  - All of Us Research Program — >680k diverse participants, whole genome sequencing (https://allofus.nih.gov/)
  - FinnGen — 500k Finnish participants with national registry linkage (https://www.finngen.fi/en)
  - dbGaP GWAS Summary Stats — thousands of published GWAS across traits (https://www.ncbi.nlm.nih.gov/gap/)
- **Starter repos/tools:**
  - SAIGE-GPU (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12960912/) — GPU-accelerated mixed-model GWAS (verify GitHub URL)
  - PRSice-2 (https://github.com/choishingwan/PRSice) — PRS computation tool (CPU; GPU via matrix backend)
  - LDpred2 (https://github.com/privefl/bigsnpr) — Bayesian PRS in R with parallelism via bigstatsr
  - PLINK 2.0 (https://www.cog-genomics.org/plink/2.0/) — genome-wide association toolkit with GPU-accelerated linear algebra
- **CUDA libraries & GPU pattern:** cuSPARSE SpMV for genotype × effect-size matrix, cuBLAS for LD matrix operations, Thrust for sorting variant effect sizes; pattern: chunked sparse matrix multiply with INT2 genotype encoding to maximise VRAM utilisation.

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
