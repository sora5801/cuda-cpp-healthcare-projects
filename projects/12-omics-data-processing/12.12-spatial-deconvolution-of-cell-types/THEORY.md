# THEORY — 12.12 Spatial Deconvolution of Cell Types

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

### 12.12 Spatial Deconvolution of Cell Types 🟡 · Active R&D
- **Deep dive:** Spatial transcriptomics spots (Visium: 55 µm, ~10 cells/spot; Visium HD: 8 µm, ~1 cell) contain mixed gene expression signals from multiple cell types; deconvolution estimates cell-type proportions per spot using a scRNA-seq reference. RCTD (Robust Cell-Type Decomposition) fits a Poisson regression per spot independently—embarrassingly parallel—enabling GPU acceleration. rctd-py achieves 9–14× GPU speedup in doublet mode and 41× in multi-cell mode on VisiumHD (~400 k spots processed in ~1 minute on a Blackwell GPU). Cell2Location uses a hierarchical Bayesian model (pyro/PyTorch) with GPU MCMC; Tangram uses optimal transport GPU acceleration.
- **Key algorithms:** Poisson regression per spot (RCTD); negative binomial regression (Cell2Location); optimal transport spot-to-reference matching (Tangram); NMF for reference-free deconvolution; stereoscope GLM; spot clustering with GPU Leiden.
- **Datasets:** 10x Genomics Visium Human Tissue datasets (https://www.10xgenomics.com/resources/datasets); Allen Brain Cell Atlas spatial data (https://portal.brain-map.org/atlases-and-data/bkp/abc-atlas); MERSCOPE public data (https://vizgen.com/data-release-program/); 4DN spatial data (https://data.4dnucleome.org/).
- **Starter repos/tools:** rctd-py (https://github.com/p-gueguen/rctd-py) — GPU-accelerated RCTD, PyTorch backend; Cell2Location (https://github.com/BayraktarLab/cell2location) — hierarchical Bayesian GPU deconvolution; Tangram (https://github.com/broadinstitute/Tangram) — OT-based GPU spatial mapping; Squidpy (https://github.com/scverse/squidpy) — spatial analysis toolkit with rapids-singlecell integration.
- **CUDA libraries & GPU pattern:** Batched Poisson regression CUDA kernels (one CUDA block per spot); PyTorch CUDA for Bayesian MCMC; GPU optimal transport (POT/GeomLoss); cuML for reference PCA; multi-GPU Dask for VisiumHD-scale spot counts.

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
