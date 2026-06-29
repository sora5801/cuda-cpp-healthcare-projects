# THEORY — 12.6 Microbiome & Antimicrobial-Resistance Analytics

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

### 12.6 Microbiome & Antimicrobial-Resistance Analytics 🟡 · Active R&D
- **Deep dive:** Microbiome profiling from shotgun metagenomics combines taxonomic classification (GPU Kraken2 / MetaCache) with functional annotation (GPU DIAMOND / MMseqs2 vs. CARD/ResFinder for AMR genes) and community ecology statistics. The AMR gene identification step—aligning millions of reads against thousands of resistance gene models (RGI uses DIAMOND + CARD)—is the most GPU-amenable component. Deep learning models (MSDeepAMR, DeepARG) trained on genomic features or mass spectrometry (MALDI-TOF) patterns predict resistance phenotypes and are accelerated by GPU inference. Metagenome-assembled genome (MAG) binning via deep learning (DAS_Tool) is also GPU-amenable.
- **Key algorithms:** K-mer-based taxonomic classification (Kraken2/MetaCache); protein homology search vs. AMR databases (DIAMOND/CARD); profile HMM search for resistance gene families; MALDI-TOF spectral CNN for phenotypic AMR prediction; random forest / gradient boosting for AMR genotype-to-phenotype; deep learning MAG binning.
- **Datasets:** CARD — Comprehensive Antibiotic Resistance Database (https://card.mcmaster.ca/); PATRIC / BV-BRC — bacterial pathogen genomes (https://www.bv-brc.org/); CAMDA AMR challenge datasets (http://www.camda.info/); HMP2 (Human Microbiome Project Phase 2) (https://www.hmpdacc.org/).
- **Starter repos/tools:** MetaCache-GPU (https://arxiv.org/pdf/2106.08150) — GPU metagenomic classifier; DIAMOND (https://github.com/bbuchfink/diamond) — GPU-targetable protein aligner for AMR annotation; DeepARG (https://github.com/gaarangoa/deeparg) — deep learning AMR gene predictor (GPU inference); RGI (https://github.com/arpcard/rgi) — Resistance Gene Identifier using CARD database.
- **CUDA libraries & GPU pattern:** GPU hash tables for k-mer AMR classification; batched cuDNN CNN inference for MALDI spectral AMR prediction; cuBLAS for alignment scoring matrix; thrust for read partition by taxon; RAPIDS cuDF for large microbiome count matrix operations.

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
