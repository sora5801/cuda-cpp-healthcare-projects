# THEORY — 4.19 Motion-Compensated 4D-CT Reconstruction

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

### 4.19 Motion-Compensated 4D-CT Reconstruction 🟡 · Active R&D
- **Deep dive:** 4D-CT captures respiratory motion by sorting ~4,000 projections into 10 breathing phases, then reconstructing each phase — effectively 10 independent 3D reconstruction problems with very few (~400) projections each (severe under-sampling). Simultaneous motion-compensated reconstruction (MCR) jointly estimates the reference volume and DVF by alternating between image reconstruction and non-rigid registration steps, each of which is a GPU-intensive computation. 4D-CBCT for adaptive radiotherapy is even more challenging (sparser projections, imaging dose constraints) and requires GPU-accelerated iterative reconstruction with motion-model regularization. Deep learning methods (4D Gaussian splatting, score-based priors) now push 4D-CBCT quality toward 4D-CT standards using GPU-trained priors.
- **Key algorithms:** Phase-binning and amplitude-binning 4D sorting, McKinnon-Bates 4D FDK, simultaneous MCR (PICCS, ROOSTER), GPU SART with deformable motion model, respiratory motion model (PCA-based surrogate), 4D neural radiance fields, 4D Gaussian splatting reconstruction.
- **Datasets:** DIR-Lab 4D-CT lung dataset (https://www.dir-lab.com/) — 10 cases with expert landmark pairs; TCIA 4D-CT lung radiotherapy collections; POPI model (https://www.creatis.insa-lyon.fr/rio/popi-model); CIRS dynamic lung phantom data.
- **Starter repos/tools:** RTK (https://github.com/RTKConsortium/RTK) — 4D ROOSTER motion-compensated reconstruction; ASTRA (https://github.com/astra-toolbox/astra-toolbox) — GPU projection kernels for 4D iterative; TIGRE (https://github.com/CERN/TIGRE) — 4D-capable iterative; Plastimatch (https://plastimatch.org/) — DIR integration with 4D dose.
- **CUDA libraries & GPU pattern:** GPU SART kernel for each phase subset; CUDA Demons for inter-phase registration; cuFFT for motion model PCA basis; texture memory for 4D DVF interpolation; alternating GPU compute between reconstruction and registration steps.

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
