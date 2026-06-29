# THEORY — 12.3 Spatial Transcriptomics Analysis

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

### 12.3 Spatial Transcriptomics Analysis 🟡 · Active R&D
- **Deep dive:** Spatial transcriptomics (10x Visium, MERFISH, Xenium) measures gene expression at spatially defined locations (thousands of spots or millions of FISH-resolved single cells), producing large dense expression × spatial matrices. GPU acceleration applies to: (1) image-based spot detection and signal decoding for MERFISH (GPU-accelerated FISH barcode decoding), (2) dimension reduction and clustering (GPU UMAP / Leiden), and (3) spatial autocorrelation statistics (Moran's I computed as a sparse matrix-vector product over spatial neighbours). A 2025 biorxiv preprint describes GPU-accelerated 3D multiplexed iterative RNA-FISH decoding, and rctd-py delivers 9–41× GPU speedup for cell-type deconvolution of Visium HD (~400 k spots).
- **Key algorithms:** FISH barcode decoding (minimum-Hamming-distance matching, GPU parallel); spatial KNN graph construction; Moran's I spatial autocorrelation (sparse MVM); NMF/NNLS for deconvolution; SpatialDE spatially variable gene regression; GPU UMAP for spatial embedding.
- **Datasets:** 10x Genomics public spatial datasets — Visium/VisiumHD human tissue (https://www.10xgenomics.com/resources/datasets); Allen Brain Cell Atlas — spatial transcriptomics of whole mouse brain (https://portal.brain-map.org/atlases-and-data/bkp/abc-atlas); 4DN spatial data portal (https://data.4dnucleome.org/); MERSCOPE (Vizgen) public datasets (https://vizgen.com/data-release-program/).
- **Starter repos/tools:** rctd-py (https://github.com/p-gueguen/rctd-py) — GPU-accelerated RCTD deconvolution, 9–41× speedup; rapids-singlecell + Squidpy integration (https://github.com/scverse/rapids_singlecell) — GPU spatial analysis; GPU-accelerated RNA-FISH decoding (https://www.biorxiv.org/content/10.1101/2025.10.10.681751.full.pdf) — 3D FISH GPU processing; Squidpy (https://github.com/scverse/squidpy) — spatial omics analysis toolkit (GPU extension via rapids-singlecell).
- **CUDA libraries & GPU pattern:** cuML UMAP / KNN for spatial graphs; cuSPARSE for spatial autocorrelation (Moran's I sparse MVM); cuDNN for FISH image decoding CNN; batched minimum-Hamming-distance kernels for MERFISH barcode matching; GPU tensor for dense spot × gene expression matrix.

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
