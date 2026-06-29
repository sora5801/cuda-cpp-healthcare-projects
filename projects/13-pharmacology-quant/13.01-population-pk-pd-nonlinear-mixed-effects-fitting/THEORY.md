# THEORY — 13.1 Population PK/PD (Nonlinear Mixed-Effects) Fitting

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

### 13.1 Population PK/PD (Nonlinear Mixed-Effects) Fitting 🟢 · Established

- **Deep dive:** Fits nonlinear mixed-effects (NLME) models to sparse individual PK/PD data from clinical trials to characterise population mean parameters (fixed effects) and between-subject variability (random effects). The computational bottleneck is the Monte Carlo EM inner loop: at each iteration, thousands of individual ODE trajectories must be integrated (one per subject per Monte Carlo sample) to compute the expected log-likelihood. GPU parallelism across subjects × MC samples provides the key speedup. A hybrid GPU-CPU implementation of parallelised MCEM for PK models (ResearchGate, 2013) demonstrated early feasibility; modern implementations using CUDA-batched RK4 ODE solvers are now standard in Pumas and experimental NONMEM backends. Each ODE evaluation for a two-compartment PK model takes ~microseconds, but millions of evaluations per EM iteration require GPU throughput.
- **Key algorithms:** FOCE (First-Order Conditional Estimation), SAEM (Stochastic Approximation EM), Laplacian approximation, Quasi-Newton BFGS optimisation, Importance Sampling for individual Bayesian estimation, Extended Kalman Filter (EKF) for continuous-time models, inter-individual variability (IIV) via log-normal random effects, correlation structures via OMEGA matrices.
- **Datasets:**
  - NONMEM Example Dataset archive — shipped with NONMEM for benchmark (verify URL via ICON plc)
  - PharmPK listserv dataset collections — published population PK datasets (verify URL)
  - CDISC SDTM/ADaM clinical trial datasets — standardised PK trial formats (https://www.cdisc.org/)
  - Warfarin PK/PD open dataset — widely used for mixed-effects benchmark (verify URL)
- **Starter repos/tools:**
  - Pumas (https://pumas.ai/) — Julia-based population PK/PD with GPU acceleration via CUDA.jl
  - Torsten (https://github.com/metrumresearchgroup/Torsten) — Stan extensions for PK/PD ODE solving with GPU potential
  - Monolix (https://lixoft.com/products/monolix/) — commercial SAEM-based NLME (verify GPU backend availability)
  - nlmixr2 (https://github.com/nlmixr2/nlmixr2) — open-source R NLME fitting with SAEM and FOCE
- **CUDA libraries & GPU pattern:** Custom CUDA RK4/RK45 batched ODE kernels, cuBLAS for OMEGA matrix operations, cuRAND for SAEM stochastic approximation draws; pattern: one CUDA thread block per subject, with inner MC samples parallelised within the block.

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
