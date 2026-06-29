# THEORY — 10.2 Real-Time Soft-Tissue Deformation for Surgical Simulation

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

### 10.2 Real-Time Soft-Tissue Deformation for Surgical Simulation 🟡 · Active R&D

- **Deep dive:** Surgical simulators require sub-10 ms deformation updates on organ meshes of tens to hundreds of thousands of elements so that haptic devices can deliver force feedback without perceived lag. Position-Based Dynamics (PBD) and its extended variant XPBD run all constraint projections in parallel, with each particle or constraint mapped to a CUDA thread. The 2024 dissection simulator demonstrated real-time performance on >100 K particles, including topological cuts, using parallelized graph-based shape matching on GPU. Material Point Method (MPM) on GPU further handles cutting and tearing by decoupling Eulerian background grids from Lagrangian particles. Hybrid organ models combining rigid bones with deformable soft tissue use adaptive octree refinement on GPU to concentrate resolution near contact zones.
- **Key algorithms:** Position-Based Dynamics (PBD/XPBD), Total Lagrangian Explicit Dynamics (TLED), graph-based shape matching, Material Point Method (MPM), corotational linear FEM, multigrid preconditioned conjugate gradient, near-second-order Jacobi/Gauss-Seidel elastodynamics (JGS2).
- **Datasets:** SOFA Framework benchmark scenes — laparoscopic and open-surgery deformable organ models (https://www.sofa-framework.org/); Kaggle Liver CT Segmentation — 3D liver meshes for deformation benchmarking (https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation); MRI Breast Tissue Segmentation (nnU-Net preprocessed) for biomechanical modeling (https://arxiv.org/abs/2411.18784); iMSTK Test Suite — pre-built surgical scenario meshes (https://www.imstk.org/).
- **Starter repos/tools:** SOFA Framework (https://github.com/sofa-framework/sofa) — open-source physics engine with GPU PBD plugins and haptic coupling; iMSTK (https://github.com/Kitware/iMSTK) — interactive medical simulation toolkit with CUDA deformation; NVIDIA FleX (https://github.com/NVIDIAGameWorks/FleX) — GPU PBD particle solver adapted for surgical contexts; CRESSim-MPM (verify URL, search "CRESSim MPM surgical simulation GPU") — GPU MPM library for cutting/suturing simulation.
- **CUDA libraries & GPU pattern:** CUDA kernels for per-constraint projection (one thread per constraint in parallel Gauss-Seidel with graph coloring), Thrust for particle neighbor search, cuSPARSE for global stiffness assembly; pattern: coloring-based Gauss-Seidel to avoid write conflicts → warp-shuffle reductions for constraint residuals → atomic updates on shared boundary nodes.

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
