# THEORY — 8.13 Vestibular System & Sensorimotor Integration

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

### 8.13 Vestibular System & Sensorimotor Integration 🔴 · Frontier/Theoretical
- **Deep dive:** The vestibular system detects head motion via semicircular canals (angular velocity → cupula deflection → hair cell activation) and otolith organs (linear acceleration). GPU simulation of the full cupula-endolymph fluid-structure interaction (FSI) in all three canals plus otolith membrane mechanics, coupled to downstream neural coding (irregular vs. regular afferents) and central vestibulo-ocular reflex (VOR) circuitry, is computationally demanding but tractable with GPU. Applications include space medicine, motion sickness modeling, and vestibular implant design.
- **Key algorithms:** Cupula-endolymph FSI (Stokes flow + elastic membrane), hair bundle adaptation ODE, afferent spike coding (van Hemmen model), torsion pendulum model, Kalman-filter Bayesian internal model, VOR motor command ODE, cerebellar Purkinje cell learning (Marr-Albus-Ito).
- **Datasets:** Vestibular electrophysiology data from DANDI (https://dandiarchive.org); Human Connectome Project functional connectivity (vestibular cortex) (https://db.humanconnectome.org); PhysioNet balance/posturography datasets (https://physionet.org); published cupula FSI experimental datasets (verify via institutional access).
- **Starter repos/tools:** NEST simulator (https://github.com/nest/nest-simulator) — vestibular afferent and VOR circuit models; GeNN (https://github.com/genn-team/genn) — GPU SNN for VOR + cerebellar learning; OpenFOAM (https://github.com/OpenFOAM/OpenFOAM-dev) — semicircular canal endolymph FSI; FEBio (https://github.com/febiosoftware/FEBio) — otolith membrane FEM.
- **CUDA libraries & GPU pattern:** Custom CUDA Stokes flow solver for endolymph; batch ODE for hair bundle + afferent dynamics (one thread per hair cell); cuBLAS for cerebellar parallel fiber weight matrix updates; pattern: fluid-structure coupling via immersed boundary method on GPU with split-step FSI.

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
