# THEORY — 12.9 GPU UMAP / t-SNE for Single-Cell Omics

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

### 12.9 GPU UMAP / t-SNE for Single-Cell Omics 🟢 · Established
- **Deep dive:** UMAP and t-SNE dimensionality reduction are the universal visualisation steps in single-cell omics (scRNA-seq, scATAC-seq, CyTOF, CITE-seq). For a million-cell dataset, standard CPU UMAP takes hours; GPU UMAP (cuML) and GPU t-SNE (RAPIDS) reduce this to minutes by parallelising the KNN graph construction (Faiss-GPU approximate nearest neighbours) and the repulsive/attractive force optimisation (each cell's gradient update is independent given the current embedding). The NVIDIA blog demonstrates GPU UMAP on 1.3 M cells processing in ~1 minute vs. ~40 minutes CPU.
- **Key algorithms:** Exact/approximate KNN (Faiss IVF-PQ, HNSWLIB-GPU); fuzzy simplicial set construction (UMAP); stochastic gradient descent with negative sampling (UMAP layout); t-SNE Barnes-Hut or FIt-SNE approximation; PCA for pre-reduction; Leiden/Louvain graph clustering.
- **Datasets:** Human Cell Atlas 10x datasets (https://www.humancellatlas.org/); CellxGene Census — 50 M+ cells (https://cellxgene.cziscience.com/); 10x Genomics 1.3 M mouse brain dataset (https://www.10xgenomics.com/resources/datasets); NCBI GEO scRNA-seq compendium (https://www.ncbi.nlm.nih.gov/geo/).
- **Starter repos/tools:** rapids-singlecell (https://github.com/scverse/rapids_singlecell) — GPU UMAP/Leiden/PCA for scRNA-seq; cuML (https://github.com/rapidsai/cuml) — GPU UMAP and t-SNE via RAPIDS; Faiss (https://github.com/facebookresearch/faiss) — GPU KNN for UMAP graph construction; NVIDIA RAPIDS single-cell examples (https://github.com/NVIDIA-Genomics-Research/rapids-single-cell-examples) — benchmarked notebooks.
- **CUDA libraries & GPU pattern:** cuML UMAP (GPU KNN + SGD layout); Faiss-GPU IVF-PQ approximate nearest neighbours; cuGraph for Leiden clustering; CUB warp-level reduction for gradient accumulation; atomic updates for asynchronous UMAP layout; multi-GPU via Dask for >10 M cells.

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
