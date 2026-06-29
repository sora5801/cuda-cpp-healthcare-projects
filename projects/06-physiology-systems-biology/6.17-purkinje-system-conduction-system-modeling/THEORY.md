# THEORY — 6.17 Purkinje System & Conduction System Modeling

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

### 6.17 Purkinje System & Conduction System Modeling 🟡 · Active R&D
- **Deep dive:** The cardiac conduction system (sinoatrial node, AV node, His bundle, bundle branches, Purkinje fiber network) initiates and coordinates ventricular activation. Simulating the Purkinje tree requires a 1D cable equation solver on a fractal branching network of ~10⁵ segments, coupled at Purkinje-muscle junctions (PMJs) to the 3D ventricular myocardium. GPU parallelism across the large number of independent 1D cable segments accelerates conduction pathway simulations for pacemaker dysfunction and re-entry arrhythmia studies.
- **Key algorithms:** 1D cable equation (monodomain) on Purkinje tree, PMJ coupling via gap-junction conductance, Stewart-Zhang Purkinje ionic model, His-Purkinje conduction velocity calibration, tree generation algorithms (L-system or rule-based branching), graph-based conduction delay computation.
- **Datasets:** openCARP community Purkinje experiments (https://opencarp.org/community/community-experiments); MonoAlg3D_C Purkinje examples (https://github.com/rsachetto/MonoAlg3D_C); NeuroMorpho (morphological analogy for tree datasets) (https://neuromorpho.org); PhysioNet His-bundle electrogram databases (https://physionet.org).
- **Starter repos/tools:** MonoAlg3D_C (https://github.com/rsachetto/MonoAlg3D_C) — GPU monodomain solver with integrated Purkinje network and PMJ calibration; openCARP (https://git.opencarp.org/openCARP/openCARP) — supports Purkinje cable coupling; Cardioid/LLNL (https://github.com/llnl/cardioid) — includes Purkinje conduction modeling; Chaste (https://github.com/Chaste/Chaste) — 1D cable equation infrastructure.
- **CUDA libraries & GPU pattern:** Batch tridiagonal solvers (cuSPARSE batched Thomas) for 1D cable segments; custom CUDA kernels for ionic ODEs at each Purkinje node; CUDA graph for recurring per-beat computation pattern; pattern: one thread per Purkinje node, shared memory for tridiagonal coefficients within a segment.

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
