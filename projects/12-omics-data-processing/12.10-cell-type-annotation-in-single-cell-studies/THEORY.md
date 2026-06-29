# THEORY — 12.10 Cell-Type Annotation in Single-Cell Studies

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

### 12.10 Cell-Type Annotation in Single-Cell Studies 🟡 · Active R&D
- **Deep dive:** Cell-type annotation assigns biological identity to each sequenced cell by comparing its gene expression profile against reference atlases or marker gene signatures. GPU acceleration applies to: (1) nearest-centroid or KNN classification in high-dimensional gene space (GPU KNN via Faiss or cuML), (2) label transfer via GPU matrix multiplication (Seurat/Harmony), and (3) foundation model inference (scGPT, Geneformer, CellMaster) that takes tokenised gene expression as input to a transformer, with GPU inference on batches of cells. scGPT (2024) fine-tuned on 33 M cells demonstrates that GPU-accelerated transformer inference at cell-type annotation is now at least as accurate as classical marker-based methods.
- **Key algorithms:** KNN label transfer in PCA-reduced gene space; Seurat anchor-based integration (CCA); marker-gene enrichment scoring (GSEA); transformer token attention over expressed genes (scGPT, Geneformer); logistic regression classifiers; hierarchical label propagation.
- **Datasets:** Human Cell Atlas (https://www.humancellatlas.org/); CellxGene Census (https://cellxgene.cziscience.com/); Azimuth reference atlases — curated cell-type references (https://azimuth.hubmapconsortium.org/); PanglaoDB — marker gene database (https://panglaodb.se/).
- **Starter repos/tools:** scGPT (https://github.com/bowang-lab/scGPT) — single-cell foundation model, GPU transformer inference; Geneformer (https://huggingface.co/ctheodoris/Geneformer) — transformer pre-trained on 30 M cells; rapids-singlecell (https://github.com/scverse/rapids_singlecell) — GPU KNN label transfer; CellMaster (https://arxiv.org/pdf/2602.13346) — collaborative annotation with LLM reasoning.
- **CUDA libraries & GPU pattern:** cuML KNN for label transfer; cuDNN transformer inference for scGPT / Geneformer; batched tokenised cell embedding via GPU; Faiss-GPU for reference atlas similarity search; multi-GPU gradient accumulation for foundation model fine-tuning.

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
