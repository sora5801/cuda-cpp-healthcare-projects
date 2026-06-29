# THEORY — 10.5 Gait & Motion-Capture Biomechanics

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

### 10.5 Gait & Motion-Capture Biomechanics 🟢 · Established

- **Deep dive:** Musculoskeletal gait analysis solves inverse kinematics (IK) and inverse dynamics (ID) to compute joint torques, followed by static optimization or forward-dynamics muscle recruitment minimizing metabolic cost. With 80+ muscles per limb and 200+ time frames per trial, the problem scales linearly with subjects in a cohort, making GPU batch-parallelism over trials the key acceleration strategy. Forward-dynamics predictive simulation using direct collocation (Moco) parallelizes across the collocation mesh nodes. GPU acceleration of Jacobian evaluation in trajectory optimization can achieve 7.7× speedup. Real-time IMU-based gait analysis on edge GPUs allows clinic-floor biomechanics without motion-capture labs.
- **Key algorithms:** Inverse kinematics (damped least-squares), inverse dynamics (Newton-Euler recursive), static optimization (bounded quadratic programming), direct collocation optimal control (Hermite-Simpson), musculotendon Hill-type models, contact detection in foot–ground models, Kalman-filter IMU fusion.
- **Datasets:** GaitRec — 2,084 patient bilateral ground reaction force (GRF) walking trials + 211 healthy controls (https://www.nature.com/articles/s41597-020-0481-z); CMU Motion Capture Database — 2500+ mocap sequences across diverse activities (http://mocap.cs.cmu.edu/); PhysioNet Gait/Posture Database — multi-camera + 17-IMU multimodal gait (https://physionet.org/content/multi-gait-posture/1.0.0/); Gait120 — comprehensive EMG + kinematic dataset (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12177048/).
- **Starter repos/tools:** OpenSim (https://github.com/opensim-org/opensim-core) — gold-standard musculoskeletal simulation; OpenSim Moco (https://github.com/opensim-org/opensim-moco) — direct collocation optimal control with multicore parallelism; Awesome-Biomechanics (https://github.com/modenaxe/awesome-biomechanics) — curated dataset/software index; PyBiomech (https://github.com/felixlb/pybiomech) — Python IMU processing pipeline (GPU-extensible).
- **CUDA libraries & GPU pattern:** cuBLAS for batch matrix inversion in IK Jacobians, Thrust for parallel over-trial static optimization QP, CUDA kernels for Hill-model force-velocity lookup tables; pattern: batch subject/trial parallelism → per-frame Jacobian assembly on GPU → CPU-side IPOPT/CasADi optimal-control solve with GPU Jacobian callbacks.

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
