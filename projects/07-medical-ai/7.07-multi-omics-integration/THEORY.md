# THEORY — 7.7 Multi-Omics Integration

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

### 7.7 Multi-Omics Integration 🟡 · Active R&D

- **Deep dive:** Combines heterogeneous molecular data layers — genomics (SNP/CNV), transcriptomics (RNA-seq), proteomics, metabolomics, and epigenomics — to predict disease subtype, drug response, or patient outcome. Integrating these layers requires jointly embedding high-dimensional sparse matrices (gene expression: 20k genes × 10k patients) with dense low-dimensional clinical vectors. GPUs accelerate the large embedding layers and transformer attention that learn cross-modal correspondences; a single multi-omics autoencoder can have hundreds of millions of parameters when modelling all layers simultaneously. scGPT-style tokenisation of omics measurements treats genes as tokens and uses CUDA-accelerated attention. Sparse input matrices benefit from cuSPARSE SpMM operations.
- **Key algorithms:** Multi-modal autoencoders (VAE, VQVAE), Graph Neural Networks over molecular interaction networks, transformer tokenisation (scGPT, mosGraphGPT), MOFA+ factor analysis, multi-task learning across omics, contrastive multi-omics pre-training, pathway-guided sparse attention.
- **Datasets:**
  - TCGA Pan-Cancer Atlas — genomic, transcriptomic, proteomic data for 33 cancer types (https://www.cancer.gov/tcga)
  - GEO (Gene Expression Omnibus) — 5M+ omics samples across species/conditions (https://www.ncbi.nlm.nih.gov/geo/)
  - CPTAC (Clinical Proteomic Tumor Analysis Consortium) — proteogenomics across tumour types (https://proteomics.cancer.gov/programs/cptac)
  - ENCODE — chromatin, transcription factor, and RNA datasets (https://www.encodeproject.org/)
- **Starter repos/tools:**
  - scGPT (https://github.com/bowang-lab/scGPT) — GPT-style multi-omics foundation model with GPU pretraining
  - MOFA+ (https://github.com/bioFAM/MOFA2) — factor analysis for multi-omics (CPU; GPU via JAX backend)
  - TF-DWGNet (https://arxiv.org/abs/2509.16301) — directed weighted GNN for multi-omics cancer subtype classification (verify URL)
  - MOLI / Concrete Autoencoder (https://github.com/mims-harvard/Madrigal) — multi-omics latent integration (verify URL)
- **CUDA libraries & GPU pattern:** cuSPARSE for sparse omics matrices, Flash Attention for gene-token sequences, NCCL multi-GPU; pattern: column-parallel embedding for gene dimension, row-parallel for sample dimension.

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
