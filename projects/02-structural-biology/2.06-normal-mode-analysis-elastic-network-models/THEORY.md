# THEORY — 2.6 Normal Mode Analysis / Elastic Network Models

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

### 2.6 Normal Mode Analysis / Elastic Network Models 🟢 · Established

- **Deep dive:** Normal Mode Analysis (NMA) computes the low-frequency vibrational modes of a protein structure, revealing collective motions (domain movements, breathing modes) relevant to allostery and function. The bottleneck is diagonalization of the 3N×3N Hessian matrix (N = atom count) — an O(N³) dense eigenvalue problem. For large proteins (N > 50,000 atoms) this is intractable on CPU. Elastic Network Models (ENMs: ANM, GNM) use simplified Hookean springs between Cα atoms, reducing the matrix but still benefiting from GPU cuSOLVER for eigendecomposition and CUDA-accelerated matrix-vector products (Lanczos iteration for sparse NMA).
- **Key algorithms:** Anisotropic network model (ANM), Gaussian network model (GNM), Hessian matrix construction (pairwise spring contacts), Lanczos/ARPACK for sparse eigendecomposition, overlap with experimental B-factors/conformational changes, RMSF from mode summation.
- **Datasets:** PDB protein structures (https://www.rcsb.org); ProDy structural dynamics dataset (https://github.com/prody/ProDy); MoDEL MD database for NMA validation (https://mmb.irbbarcelona.org/MoDEL/); flexnMR NMR flexibility benchmark (verify URL).
- **Starter repos/tools:** ProDy (https://github.com/prody/ProDy) — Python NMA/ENM with GPU support via PyTorch; iModS server (https://imods.iqfr.csic.es) — NMA-based motion analysis; Bio3D R package (https://thegrantlab.org/bio3d/) — NMA in R; ElNemo (https://www.sciences.univ-nantes.fr/elnemo/) — elastic network modes server.
- **CUDA libraries & GPU pattern:** cuSOLVER dense dsyevd for moderate-sized Hessians; cuSPARSE for sparse ANM matrix-vector products; custom CUDA Lanczos iteration for large sparse NMA; cuBLAS for B-factor RMSF accumulation.

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
