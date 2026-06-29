# THEORY — 13.8 Therapeutic Dose Individualisation / Model-Informed Dosing

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

### 13.8 Therapeutic Dose Individualisation / Model-Informed Dosing 🟡 · Active R&D

- **Deep dive:** Adapts drug dosing for individual patients using Bayesian updating of a population PK/PD prior with the patient's own concentration measurements (therapeutic drug monitoring, TDM). GPU acceleration is relevant in three ways: (1) population model fitting on GPU (as in 13.1); (2) real-time posterior ODE integration for thousands of candidate dose levels simultaneously to find the optimal dose; (3) simulation-based model averaging across uncertainty in individual parameters. The AUC-target dosing problem reduces to: for each candidate dose schedule, integrate the PK ODE forward for 30 days and check whether AUC hits target — parallelised across doses on GPU. Pumas and Bayesian NONMEM implement this on GPU.
- **Key algorithms:** Bayesian individual parameter estimation (MAP, full Bayes), AUC-target optimisation via GPU-parallel ODE forward simulation, MAP-adaptive dosing, Model Predictive Control (MPC) for infusion rate optimisation, optimal sampling time selection (D-optimal), individual dose prediction with uncertainty propagation, neural ODE for personalised PK.
- **Datasets:**
  - Published TDM datasets (vancomycin, aminoglycosides, tacrolimus) — available through PharmPK listserv (verify URL)
  - NONMEM example datasets — shipped with NONMEM installation (verify URL)
  - Latent Neural-ODE paper dataset (https://arxiv.org/abs/2602.03215) — personalised dosing with neural ODE
  - MIMIC-IV medication and lab data — vancomycin AUC retrospective cohorts (https://physionet.org/content/mimiciv/)
- **Starter repos/tools:**
  - Pumas (https://pumas.ai/) — GPU-accelerated Bayesian dose individualisation in Julia
  - Torsten (https://github.com/metrumresearchgroup/Torsten) — Stan extension for PK ODE solving; Bayesian TDM
  - InsightRx (verify URL) — commercial Bayesian dosing platform
  - BayesPK (verify URL) — open-source Bayesian PK software for TDM
- **CUDA libraries & GPU pattern:** Custom CUDA ODE kernels for forward simulation across dose grid, cuRAND for uncertainty sampling, Thrust for AUC computation; pattern: dose-grid-parallel ODE integration — each CUDA thread evaluates one dose schedule forward simulation.

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
