# THEORY — 10.16 Surgical Robot Path Planning & Collision Detection

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

### 10.16 Surgical Robot Path Planning & Collision Detection 🟡 · Active R&D

- **Deep dive:** Robotic-assisted surgery (e.g., da Vinci, Mako) requires real-time collision-free trajectories for multiple articulated arms moving near deformable anatomy. GPU parallel motion planning (RRT*, PRM) checks thousands of configuration-space samples for collision against a GPU-resident signed-distance-field (SDF) of the patient anatomy simultaneously, achieving path generation in under 100 ms — 50–100× faster than CPU planners. Deep-learning collision detectors trained in simulation (Learning-from-Simulation, 2025) replace explicit geometric checks with GPU neural networks, handling soft-tissue deformation that classical rigid-body checkers cannot. The GPU also runs online model-predictive controllers that re-plan at 50 Hz as tissue moves during respiration.
- **Key algorithms:** GPU-parallel RRT*/PRM with SDF collision query, signed-distance-field generation via GPU ray marching, neural collision detector (implicit neural representation), MPC for force-controlled insertion, generalized momentum observer for external-force estimation.
- **Datasets:** SurgRobotics Dataset — da Vinci tool tracking + anatomy meshes (verify URL via MICCAI); SCARED Dataset — stereo depth reconstruction in laparoscopy (https://endovissub2019-scared.grand-challenge.org/); MICCAI 2024 Surgical Scene Segmentation Challenge (verify URL via Grand Challenge); CholecT50 (https://github.com/CAMMA-public/cholect50) — tool-tissue interaction labels.
- **Starter repos/tools:** GPU-based Parallel Collision Detection (UNC Gamma group, http://gamma.cs.unc.edu/gplanner/) — GPU PRM reference; cuRobo (https://github.com/NVlabs/curobo) — NVIDIA CUDA-accelerated robot motion generation; SOFA Framework (https://github.com/sofa-framework/sofa) — deformable anatomy + robot coupling; IsaacGym (https://developer.nvidia.com/isaac-gym) — GPU parallel surgical-robot RL training.
- **CUDA libraries & GPU pattern:** CUDA kernels for SDF generation (parallel ray marching), cuDNN for neural collision network inference, Thrust for parallel RRT sample feasibility checks; pattern: GPU SDF updated from tissue deformation → 4096 configuration samples checked in parallel → feasible path selected → MPC re-plan at 50 Hz → torque commands dispatched.

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
