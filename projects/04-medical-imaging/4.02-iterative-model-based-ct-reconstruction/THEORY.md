# THEORY — 4.2 Iterative / Model-Based CT Reconstruction

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

### 4.2 Iterative / Model-Based CT Reconstruction 🟡 · Active R&D
- **Deep dive:** Instead of a single analytical inversion, iterative methods repeatedly forward-project a current volume estimate, compare to measured sinogram data, then backproject the residual with statistical weighting. Penalized weighted least squares (PWLS) with total-variation (TV) or dictionary priors reduces noise by 30–50% at matched dose compared with FBP. Each outer iteration performs one full forward-projection and one backprojection — exactly the same GPU kernel bottleneck as FBP but repeated 20–200 times, making GPU mandatory for clinical throughput. ADMM decouples the data-fidelity and regularization sub-problems, enabling efficient GPU-friendly matrix-vector operations. Statistical models (Poisson likelihood for photon counts) can be incorporated for dose-optimal reconstruction.
- **Key algorithms:** SIRT, SART, OS-EM for CT, PWLS-TV, PWLS with dictionary/wavelet priors, ADMM, primal-dual splitting (Chambolle-Pock), model-based iterative reconstruction (MBIR), plug-and-play ADMM with DnCNN denoiser.
- **Datasets:** 2016 AAPM Low-Dose CT Grand Challenge (https://www.aapm.org/grandchallenge/lowdosect/); Mayo Clinic Low-Dose CT dataset (available via TCIA); LIDC-IDRI via TCIA (https://www.cancerimagingarchive.net/).
- **Starter repos/tools:** ASTRA Toolbox (https://github.com/astra-toolbox/astra-toolbox) — GPU primitives, build iterative loops in Python/MATLAB; TIGRE (https://github.com/CERN/TIGRE) — includes OS-TV, SART, CGLS with GPU acceleration; ODL (Operator Discretization Library, https://github.com/odlgroup/odl) — Python framework wrapping ASTRA for variational reconstruction; LEAP (https://github.com/LLNL/LEAP) — LLNL GPU-accelerated CT reconstruction library with penalized-likelihood support.
- **CUDA libraries & GPU pattern:** cuSPARSE (sparse system matrix), cuFFT, custom CUDA kernels for voxel-driven projection; outer loop on CPU, inner GPU kernel per OS subset; shared-memory tile reuse for cone-beam geometry.

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
