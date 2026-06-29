# THEORY — 6.19 Defibrillation & High-Voltage Shock Simulation

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

### 6.19 Defibrillation & High-Voltage Shock Simulation 🟡 · Active R&D
- **Deep dive:** Defibrillation delivers a high-voltage electric field across the myocardium to terminate ventricular fibrillation. Simulating shock efficacy requires solving the bidomain equations driven by extracellular electrode currents, capturing virtual electrode polarization (VEP)—regions of depolarization and hyperpolarization induced at tissue boundaries—and subsequent re-entry termination. The nonlinear ionic response during shock (10 V/cm field, sub-ms timescale) and the fine spatial resolution needed (~0.1 mm) make GPU acceleration mandatory for whole-heart shock simulations.
- **Key algorithms:** Bidomain equations with extracellular stimulus, virtual electrode polarization theory, finite volume/element discretization, operator splitting with Rush-Larsen ionic integration, conjugate gradient linear solver, shock-protocol optimization (monophasic vs. biphasic), defibrillation threshold (DFT) estimation.
- **Datasets:** PhysioNet fibrillation/defibrillation recordings (https://physionet.org); openCARP defibrillation tutorial cases (https://opencarp.org); Cardioid (https://github.com/llnl/cardioid) — bidomain shock examples; patient-specific ICD placement datasets (verify institutional access).
- **Starter repos/tools:** openCARP (https://git.opencarp.org/openCARP/openCARP) — bidomain solver with extracellular stimulus for defibrillation studies; MonoAlg3D_C (https://github.com/rsachetto/MonoAlg3D_C) — GPU bidomain-capable extension; Cardioid/LLNL (https://github.com/llnl/cardioid) — cardiac EP + shock; Chaste (https://github.com/Chaste/Chaste) — bidomain with electrode boundary conditions.
- **CUDA libraries & GPU pattern:** cuSPARSE conjugate gradient for bidomain elliptic solve; custom CUDA kernels for per-cell ionic ODE during shock timescale (0.01 ms dt); CUDA Unified Memory for large torso+heart mesh; pattern: dual-grid approach—fine heart mesh on GPU, coarse torso on CPU, coupled via interface boundary.

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
