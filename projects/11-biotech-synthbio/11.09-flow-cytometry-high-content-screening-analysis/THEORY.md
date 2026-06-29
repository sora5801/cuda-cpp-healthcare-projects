# THEORY — 11.9 Flow Cytometry & High-Content Screening Analysis

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

### 11.9 Flow Cytometry & High-Content Screening Analysis 🟡 · Active R&D

- **Deep dive:** Modern cell sorters generate 10⁶ cells/second at 20–50 parameters per event; high-content screening (HCS) platforms image millions of cells per plate with 10+ channels. GPU-accelerated dimensionality reduction (GPU-UMAP, GPU-TSNE via RAPIDS cuML) and clustering (GPU-HDBSCAN, GPU-PhenoGraph) turn 30-minute analyses into seconds, enabling real-time sort gates. GPU-accelerated CellProfiler-style morphological feature extraction processes 96-well plate images in minutes instead of hours. Deep-learning classifiers (ResNet, ViT) deployed on GPU identify rare phenotypes (1-in-10⁵ events) with high sensitivity.
- **Key algorithms:** GPU-UMAP (approximate nearest-neighbor with NN-descent), GPU-HDBSCAN, GPU FlowSOM self-organizing map, GPU PhenoGraph graph-based clustering, GPU CellPose segmentation for HCS, Wasserstein distance for batch-effect correction, GPU deep learning rare-event classifier.
- **Datasets:** FlowRepository — public flow cytometry FCS files (https://flowrepository.org/); JUMP-CP — 116 K compound HCS morphological profiles, RxRx cell-painting images (https://jump-cellpainting.broadinstitute.org/); Cell Painting Gallery (Broad Institute) — 140 TB cell images (https://registry.opendata.aws/cellpainting-gallery/); Human Protein Atlas imaging (https://www.proteinatlas.org/).
- **Starter repos/tools:** RAPIDS cuML (https://github.com/rapidsai/cuml) — GPU UMAP/TSNE/HDBSCAN for cytometry analysis; CellProfiler (https://github.com/CellProfiler/CellProfiler) — HCS morphological profiling (with GPU CellPose segmentation); CellPose (https://github.com/mouseland/cellpose) — GPU-accelerated cell segmentation; FlowKit (https://github.com/whitews/FlowKit) — FCS file processing (CPU; upstream of GPU analysis).
- **CUDA libraries & GPU pattern:** cuML GPU-UMAP, cuDNN for ResNet cell image classifier, CUDA 2D convolution kernels for morphological feature extraction; pattern: FCS/image batch ingest → GPU feature extraction → GPU-UMAP embedding → GPU-HDBSCAN clustering → rare-event gating → real-time sort decisions.

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
