# THEORY — 3.12 Single-Cell RNA-seq Analysis

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

### 3.12 Single-Cell RNA-seq Analysis 🟡 · Active R&D
- **Deep dive:** Single-cell RNA-seq (scRNA-seq) produces count matrices for tens of millions of cells × 30 k genes; downstream analysis involves normalisation, highly variable gene selection, PCA, k-nearest-neighbour graph construction (O(n²) naive, accelerated by approximate nearest neighbours), UMAP / t-SNE embedding, Leiden/Louvain clustering, and differential expression. rapids-singlecell (scverse, 2024) replaces Scanpy's NumPy/SciPy backend with cuPy, cuML, and cuGraph equivalents, achieving >20× speedup for datasets up to 20 M cells. The KNN graph construction and UMAP optimisation are the most GPU-impactful steps, turning hours into minutes.
- **Key algorithms:** Normalised count transformation (scran/Seurat); PCA on sparse count matrix; approximate KNN (Faiss, HNSWLIB GPU); UMAP force-directed layout; Leiden graph clustering; negative binomial GLM for differential expression; doublet detection.
- **Datasets:** Human Cell Atlas — multi-organ scRNA-seq compendium (https://www.humancellatlas.org/); 10× Genomics public datasets (https://www.10xgenomics.com/resources/datasets); CellxGene Census — 50 M+ cells (https://cellxgene.cziscience.com/); NCBI GEO — thousands of scRNA-seq studies (https://www.ncbi.nlm.nih.gov/geo/).
- **Starter repos/tools:** rapids-singlecell (https://github.com/scverse/rapids_singlecell) — drop-in GPU Scanpy replacement, cuPy/cuML/cuGraph; NVIDIA RAPIDS single-cell examples (https://github.com/NVIDIA-Genomics-Research/rapids-single-cell-examples) — benchmark notebooks up to 1 M+ cells; ScaleSC (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12321287/) — GPU scRNA pipeline, 20× speed, 20 M cells on A100; Scanpy (https://github.com/scverse/scanpy) — CPU reference with GPU-aware backends.
- **CUDA libraries & GPU pattern:** cuPy sparse GEMM (count matrix ops); cuML PCA, UMAP, KNN; cuGraph Leiden/Louvain; Faiss-GPU HNSW index for ANN; cuDF for dataframe operations; multi-GPU Dask for datasets exceeding GPU RAM.

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
