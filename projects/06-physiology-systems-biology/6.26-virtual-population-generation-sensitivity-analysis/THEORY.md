# THEORY — 6.26 Virtual Population Generation & Sensitivity Analysis

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

### 6.26 Virtual Population Generation & Sensitivity Analysis 🟡 · Active R&D
- **Deep dive:** Virtual patient populations are created by sampling physiological parameter distributions (body weight, organ volumes, enzyme expression, sex, age) from measured databases (NHANES, WHO) and propagating them through PBPK/PD models to generate simulated trial cohorts. Sobol sensitivity analysis requires O(N×(2k+2)) model evaluations for k parameters—typically millions of forward ODE integrations. GPU batch simulation reduces this from days to hours.
- **Key algorithms:** Latin hypercube sampling (LHS), Sobol quasi-random sequences, Morris one-at-a-time elementary effects, Sobol variance-based sensitivity indices, polynomial chaos expansion (PCE), Gaussian process surrogate (emulator), MCMC parameter estimation (Metropolis-Hastings, NUTS), bootstrap confidence intervals.
- **Datasets:** NHANES anthropometric/physiological data (https://www.cdc.gov/nchs/nhanes/); WHO growth reference datasets (https://www.who.int/tools/growth-reference-data-for-5to19-years); OSP PBPK model library (https://github.com/Open-Systems-Pharmacology/OSP-PBPK-Model-Library); FDA drug label PK data (https://www.fda.gov/drugs).
- **Starter repos/tools:** SALib sensitivity analysis library (https://github.com/SALib/SALib) — Morris, Sobol, FAST methods for Python; Open Systems Pharmacology (https://github.com/Open-Systems-Pharmacology) — virtual population creation module (PK-Sim); mrgsolve (https://github.com/metrumresearchgroup/mrgsolve) — fast ODE PK batch simulation; SUNDIALS batch CVODE (https://github.com/LLNL/sundials) — GPU ODE ensemble.
- **CUDA libraries & GPU pattern:** cuRAND for Sobol/Halton quasi-random sequences; batch CVODE GPU for ensemble ODE; cuBLAS for PCE coefficient matrix operations; pattern: one CUDA thread per virtual patient, Sobol sensitivity via GPU-parallel model evaluations; thrust::transform for per-sample output extraction.

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
