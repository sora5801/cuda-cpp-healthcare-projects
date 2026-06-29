# THEORY — 6.10 Systems-Biology ODE/SDE Network Solver

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

### 6.10 Systems-Biology ODE/SDE Network Solver 🟡 · Active R&D
- **Deep dive:** Gene regulatory networks, signaling cascades, and metabolic models are encoded as systems of potentially thousands of nonlinear ODEs/SDEs (e.g., SBML models from BioModels). Integrating a single model is fast, but parameter sweeps, uncertainty quantification, and multi-cell applications require solving thousands of independent instances simultaneously—a perfectly GPU-parallel batch problem. SUNDIALS/CVODE-GPU and libRoadRunner's LLVM JIT backend both target this batch-ODE pattern.
- **Key algorithms:** CVODE adaptive BDF/Adams multistep integrator, explicit Euler / Runge-Kutta (RK4, RK45 Dormand-Prince) for stiff-moderate systems, implicit trapezoidal, chemical Langevin equation (CLE) for SDE, sensitivity equations (CVODES/IDAS), SBML parsing and JIT compilation.
- **Datasets:** BioModels Database (EMBL-EBI) — 1000+ curated SBML models (https://www.ebi.ac.uk/biomodels); Reactome pathways — curated molecular interaction data (https://reactome.org); BioGRID interaction network (https://thebiogrid.org); VCell curated models (https://vcell.org).
- **Starter repos/tools:** SUNDIALS/CVODE GPU (https://github.com/LLNL/sundials) — LLNL ODE/DAE solver with CUDA NVector and GPU-accelerated batch CVODE; libRoadRunner (https://github.com/sys-bio/roadrunner) — high-performance SBML ODE integrator with LLVM JIT, GPU batch mode in development; Tellurium (https://github.com/sys-bio/tellurium) — Python systems biology platform built on roadrunner; GillesPy2 (https://github.com/GillesPy2/GillesPy2) — SSA + tau-leaping + CLE stochastic solver.
- **CUDA libraries & GPU pattern:** CUDA batched ODE: one CUDA thread-block per ODE system; shared memory for Jacobian; cuSPARSE for large sparse Jacobians; SUNDIALS CUDA NVector; pattern: batch-CVODE with user-supplied CUDA right-hand-side (RHS) kernel.

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
