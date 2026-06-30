# Push 2026-06-30 #04 -- phase2 batch4b genomics intermediate

> Push-note (CLAUDE.md section 7.1). Second domain-3 batch: the last Beginner + first 5
> Intermediate genomics projects, each worker-built and lead-verified.

## 1. Summary

Six more **domain-3 (genomics)** projects are complete, taking the collection to
**88 -> 94 / 301 (31.2%)** and domain 3 to **13/30**. This batch finishes the Beginner tier
(BQSR) and opens the Intermediate tier with the field's heavy hitters: de novo assembly
overlap, multiple sequence alignment, phylogenetic likelihood, RNA secondary structure, and
GWAS at scale. It introduces a **new CUDA library to the collection — cuBLAS** (GWAS
relatedness via DGEMM) — and showcases two classic DP-on-GPU shapes (O(N^2) pairwise NW, the
anti-diagonal **wavefront** for Nussinov). Each was built in its own folder by one worker and
re-verified by the lead.

## 2. What changed

Six new fully-implemented projects under `projects/03-genomics/`:

- [`3.25` Base Quality Score Recalibration (BQSR)](../projects/03-genomics/3.25-base-quality-score-recalibration-bqsr)
- [`3.05` De Novo Genome Assembly](../projects/03-genomics/3.05-de-novo-genome-assembly)
- [`3.08` Multiple Sequence Alignment (MSA)](../projects/03-genomics/3.08-multiple-sequence-alignment-msa)
- [`3.09` Phylogenetic Likelihood / Tree Inference](../projects/03-genomics/3.09-phylogenetic-likelihood-tree-inference)
- [`3.10` RNA Secondary-Structure Prediction](../projects/03-genomics/3.10-rna-secondary-structure-prediction)
- [`3.11` GWAS at Scale](../projects/03-genomics/3.11-gwas-at-scale)

`docs/STATUS.md` -> these 6 marked **done** (94/301). `CHANGELOG.md` indexed.

## 3. New projects (didactic blurb each)

- **3.25 BQSR** — Base Quality Score Recalibration: a covariate-table scan (one thread per
  base, integer atomicAdd into (Q, cycle, dinuc-context) bins) with known-variant masking +
  empirical-quality recalibration. Shared `bqsr.h` core -> exact CPU==GPU (reads reported Q30
  recover Q_emp=19). The GATK BQSR step, in miniature.
- **3.05 De Novo Assembly (overlap)** — the all-vs-all read-overlap stage: per-read
  **minimizer sketches** + one-thread-per-read-pair shared-minimizer counting over a flat CSR
  layout, shared merge-intersection core -> exact CPU==GPU. Recovers the expected single contig
  from a synthetic 6-read FASTA. The overlap engine behind OLC assemblers.
- **3.08 MSA** — the O(N^2) **pairwise Needleman-Wunsch** distance phase, one block per pair
  (shared `nw_core.h`, exact CPU==GPU score matrices), then host **center-star + progressive**
  assembly into the final alignment. A scaled-up cousin of the Smith-Waterman flagship.
- **3.09 Phylogenetic Likelihood** — **Felsenstein's pruning** recursion under the K2P model,
  one thread per alignment site (tree in constant memory, deterministic fixed-point atomic
  reduction). Exact CPU==GPU; the ML winner is the true simulated tree. The compute core of
  RAxML/IQ-TREE.
- **3.10 RNA Secondary Structure** — **Nussinov** max-base-pair DP via the **anti-diagonal
  wavefront** pattern (one kernel launch per span), shared recurrence -> bit-identical integers.
  The 18-nt hairpin folds to the known "((((((....))))))..". The wavefront DP, applied to RNA.
- **3.11 GWAS at Scale** — **cuBLAS DGEMM** genetic-relatedness matrix GRM = (1/M)ZZ^T plus a
  one-thread-per-SNP single-marker regression scan; shared math core -> CPU/GPU agree ~1e-16
  (GRM) / ~1e-13 (chi^2). All 5 injected causal SNPs recovered as the top 5 hits. **First
  cuBLAS project in the collection.**

All six are clearly-labeled **reduced-scope teaching versions** (synthetic reads/cohorts,
labeled synthetic), with production tools (GATK BQSR, Hifiasm/Canu, MAFFT/MUSCLE, RAxML/
IQ-TREE, ViennaRNA, PLINK/GCTA/REGENIE) named in each `THEORY.md`.

## 4. How to build & run

```powershell
cd projects/03-genomics/3.11-gwas-at-scale   # (or any of the six)
msbuild build/*.sln /p:Configuration=Release /p:Platform=x64
./demo/run_demo.ps1      # -> RESULT: PASS (GPU matches CPU)
```

3.11 links **cuBLAS** (`cublas.lib` in both `<Link>` sections + `CMakeLists.txt`,
BUILD_GUIDE §7b). The others are pure custom kernels (integer DP, atomics, CSR).

## 5. What to study here

Reading path: **3.25** (covariate-table atomics) -> **3.05** (minimizer overlap) -> **3.08**
(pairwise NW, one block per pair) -> **3.10** (the anti-diagonal wavefront DP) -> **3.09**
(Felsenstein pruning over a tree) -> **3.11** (cuBLAS GEMM + a per-SNP scan). The three DP
projects (3.08, 3.09, 3.10) plus flagship 3.01 are a mini-course in "dynamic programming on
the GPU". Exercise: in **3.10**, lengthen the RNA and watch the wavefront launch count grow
as 2N-1; in **3.11**, raise the number of causal SNPs and confirm they still top the scan.

## 6. Verification (lead-independent, not self-reports)

- ✅ **Boundaries:** only the 6 project folders changed; no shared/root file; no artifacts.
- ✅ **Clean rebuild** (`/t:Rebuild`, fat arch list) of all 6 in both `Release|x64` and
  `Debug|x64`: **EXIT=0, 0 warnings, 0 errors** (12/12 builds), incl. the cuBLAS link in 3.11.
- ✅ All 6 **demos PASS**: GPU==CPU (assembly/MSA/phylo/RNA/BQSR exact; GWAS ~1e-13).
- ✅ `verify_project.py` -> **DONE** for all 6 (comment ratios **0.78–1.02**).
- **Workflow:** 6 agents, ~1.06M agent tokens, 441 tool uses.
- **Environment:** RTX 2080 (SUPER), `sm_75`, CUDA 13.3, VS 2026 (`v145`).

## 7. Known limitations / TODOs

- All six are **reduced-scope teaching versions**: an overlap stage (not full assembly), a
  small MSA family, a few candidate trees, short RNAs, a small synthetic cohort. Labeled
  synthetic; production scale described in each THEORY.md.
- **PATTERNS.md** should gain a cuBLAS GEMM entry now that 3.11 uses it (tracked for a near-term
  docs touch-up).

## 8. Next push preview

Continue domain-3 Intermediates (`3.12` scRNA-seq, `3.13` pangenome alignment, `3.14`
metagenomic classification, `3.15` Hi-C, `3.16` error correction, `3.17` CRISPR guide design,
…) in ~6-project batches through to `3.30`. Same workflow, lead-verified, one push-note per batch.
