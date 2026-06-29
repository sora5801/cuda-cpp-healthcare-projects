# THEORY — 6.15 PK/PD & PBPK Modeling

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

### 6.15 PK/PD & PBPK Modeling 🟢 · Established
- **Deep dive:** Pharmacokinetic/pharmacodynamic (PK/PD) and physiologically-based PK (PBPK) models are compartmental ODE systems describing drug absorption, distribution, metabolism, and excretion across tissues. Population PK analysis (NLME) requires solving the ODE model for each individual in a cohort (hundreds to thousands) with Monte Carlo sampling of parameter distributions—perfectly GPU-parallel. GPU speedup reaches 10–100× for population-level stochastic simulations and Bayesian posterior sampling (HMC/NUTS).
- **Key algorithms:** Compartmental ODE integration (1-cpt, 2-cpt, PBPK), nonlinear mixed-effects (NLME) estimation, empirical Bayes estimation (EBE), Monte Carlo simulation, Bayesian MCMC (Hamiltonian Monte Carlo, NUTS), sensitivity analysis (Morris screening, Sobol indices), indirect-response PD models, transit compartment absorption.
- **Datasets:** PhysioNet MIMIC clinical PK data (https://physionet.org); FDA Adverse Event Reporting System (FAERS) (https://www.fda.gov/drugs/fda-adverse-event-reporting-system-faers); PBPK model library — OSP Suite (https://github.com/Open-Systems-Pharmacology/OSP-PBPK-Model-Library); DDMoRe model repository (https://ddmore.eu/models-tools).
- **Starter repos/tools:** Open Systems Pharmacology Suite (https://github.com/Open-Systems-Pharmacology) — PK-Sim + MoBi PBPK platform; mrgsolve (https://github.com/metrumresearchgroup/mrgsolve) — R-based ODE PK/PD simulation; Pumas-AI (https://pumas.ai) — Julia pharmacometrics platform with GPU-accelerated population PK; GillesPy2 (https://github.com/GillesPy2/GillesPy2) — stochastic PK variant simulation.
- **CUDA libraries & GPU pattern:** SUNDIALS batch CVODE on GPU (population member = one GPU batch element); cuRAND for Monte Carlo parameter sampling; custom CUDA kernel for NLME gradient (sum over individuals); pattern: one CUDA thread per subject for ODE integration, warp-level reduction for population log-likelihood.

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
