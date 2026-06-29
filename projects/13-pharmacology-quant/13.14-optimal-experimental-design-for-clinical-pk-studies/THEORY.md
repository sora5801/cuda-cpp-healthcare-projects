# THEORY — 13.14 Optimal Experimental Design for Clinical PK Studies

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

### 13.14 Optimal Experimental Design for Clinical PK Studies 🔴 · Frontier/Theoretical

- **Deep dive:** Identifies optimal blood sampling times and dose levels for population PK studies to maximise Fisher Information (D-optimality) or minimise cost given constraints on sample number and patient burden. The Fisher Information Matrix (FIM) for a nonlinear mixed-effects model requires integrating ODE trajectories at all candidate sampling times and evaluating the sensitivity of outputs to parameters — an O(N_times × N_params²) computation. GPU parallelism across candidate design grids (millions of combinations of sampling times × doses) enables global search in hours vs. days. Bayesian optimisation of design using surrogate models trained on GPU-simulated FIM evaluations represents the frontier approach.
- **Key algorithms:** D-optimal, A-optimal, E-optimal design criteria on Fisher Information Matrix, MFIM (Matrix FIM) computation for NLME, Bayesian D-optimality (BOED), Sequential Bayesian Experimental Design, derivative-informed neural operators for FIM surrogate, population FIM via importance sampling, D-optimal dose selection for Phase I.
- **Datasets:**
  - NONMEM example datasets for FIM validation (verify URL)
  - PopED example models — software-integrated benchmark designs (https://github.com/andrewhooker/PopED)
  - PAGE (Population Approach Group in Europe) OED workshop data (verify URL)
  - Optimal experiment design PK examples (https://pmc.ncbi.nlm.nih.gov/articles/PMC11996619/)
- **Starter repos/tools:**
  - PopED (https://github.com/andrewhooker/PopED) — R/MATLAB optimal experimental design for population PK
  - PFIM (verify URL) — R package for Fisher Information Matrix-based design
  - Pumas OptimalDesign extension (https://pumas.ai/) — GPU-accelerated OED in Julia
  - Pyomo.DoE (https://github.com/IDAES/idaes-pse) — Python optimal experimental design (verify URL)
- **CUDA libraries & GPU pattern:** Custom CUDA sensitivity ODE kernels for FIM computation, cuBLAS for FIM matrix determinant (log-det D-criterion), cuRAND for Bayesian design Monte Carlo; pattern: GPU grid search over sampling time combinations with parallel FIM evaluation per design point.

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
