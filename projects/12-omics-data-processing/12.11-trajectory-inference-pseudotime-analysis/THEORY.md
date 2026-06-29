# THEORY — 12.11 Trajectory Inference & Pseudotime Analysis

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

### 12.11 Trajectory Inference & Pseudotime Analysis 🟡 · Active R&D
- **Deep dive:** Trajectory inference reconstructs continuous developmental processes from snapshot scRNA-seq data by ordering cells along a pseudotime axis representing biological progression (differentiation, cell cycle, immune activation). Algorithms range from principal curve fitting (Monocle3) to diffusion-map graph-based approaches (Scanpy PAGA) and optimal transport (Waddington-OT). GPU acceleration targets the KNN graph construction (the shared first step), the diffusion map eigensolver (cuSolver), and the optimal transport (Sinkhorn algorithm) computation. For atlas-scale data (>1 M cells), GPU trajectory inference with RAPIDS reduces hours to minutes.
- **Key algorithms:** Principal curve / elastic principal graph (DDRTree); diffusion pseudotime (DPT) via diffusion map eigenvectors; PAGA graph abstraction; RNA velocity (scVelo) splicing dynamics EM; Sinkhorn optimal transport for fate probability; graph-based geodesic distances for branch assignment.
- **Datasets:** Human Cell Atlas developmental atlases (https://www.humancellatlas.org/); GEO scRNA-seq differentiation time-course datasets (https://www.ncbi.nlm.nih.gov/geo/); Allen Brain Cell Atlas (https://portal.brain-map.org/atlases-and-data/bkp/abc-atlas); ENCODE iPSC differentiation scRNA-seq (https://www.encodeproject.org/).
- **Starter repos/tools:** rapids-singlecell (https://github.com/scverse/rapids_singlecell) — GPU diffusion pseudotime and UMAP; Scanpy with GPU backend (https://github.com/scverse/scanpy) — PAGA trajectory analysis; scVelo (https://github.com/theislab/scvelo) — RNA velocity (GPU EM target); Monocle3 (https://github.com/cole-trapnell-lab/monocle3) — principal graph trajectory inference.
- **CUDA libraries & GPU pattern:** cuSolver eigenvalue solver for diffusion map; cuSPARSE for KNN graph Laplacian operations; cuML for PCA pre-reduction; custom Sinkhorn CUDA kernels (iterative row/column normalisation); GPU optimal transport via POT library with CUDA backend.

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
