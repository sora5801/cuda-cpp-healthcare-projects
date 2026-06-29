# THEORY — 4.21 MR Fingerprinting Reconstruction

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

### 4.21 MR Fingerprinting Reconstruction 🟡 · Active R&D
- **Deep dive:** MR Fingerprinting (MRF) acquires a sequence of images with pseudorandom flip angles and TRs; each voxel's signal time course is matched to a dictionary of simulated Bloch-equation evolutions to simultaneously estimate T1, T2, and other parameters. The dictionary (10⁵–10⁶ entries × 1,000 time points) must be searched for each of ~10⁵ voxels, resulting in ~10¹¹ inner products — efficiently computed as a single large matrix-matrix product on GPU (cuBLAS GEMM). Compressed MRF combines partial k-space acquisition with low-rank tensor reconstruction, reducing the GPU workload to manageable batches. Non-Cartesian MRF trajectories require NUFFT-based reconstruction, adding a cuFFT step.
- **Key algorithms:** Bloch-simulation dictionary generation, dot-product template matching (inner product per voxel per dictionary entry as GEMM), low-rank subspace reconstruction, ADMM+MRF, physics-driven neural network MRF (DeepMRF), sequence optimization via Cramér-Rao bound.
- **Datasets:** fastMRI MRF (verify URL at fastmri.org); Cleveland Clinic MRF dataset (via IEEE DataPort, verify URL); synthetic MRF datasets generated from XCAT/BrainWeb phantoms; public multi-parametric MRI from qMRI.org (verify URL).
- **Starter repos/tools:** BART (https://github.com/mrirecon/bart) — low-rank subspace MRF reconstruction; MRzero (https://github.com/MRsimulator/MRzero) — differentiable MR sequence simulation for MRF design; PyTorch MRF dictionary matching (search GitHub for "MRF dictionary matching PyTorch"); SigPy (https://github.com/mikgroup/sigpy) — NUFFT-based MRF reconstruction.
- **CUDA libraries & GPU pattern:** cuBLAS SGEMM for dictionary matching (entire voxel×time matrix vs. dictionary×time matrix); cuFFT for NUFFT in non-Cartesian MRF; GPU-pinned memory for dictionary transfer; batched GEMM across slices via cuBLAS-Xt.

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
