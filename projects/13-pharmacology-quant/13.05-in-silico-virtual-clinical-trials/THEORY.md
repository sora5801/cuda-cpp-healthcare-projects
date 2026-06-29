# THEORY — 13.5 In Silico Virtual Clinical Trials

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

### 13.5 In Silico Virtual Clinical Trials 🟡 · Active R&D

- **Deep dive:** Generates virtual patient populations and runs complete simulated clinical trials in silico to optimise dose, schedule, and eligibility criteria before committing to expensive Phase II/III studies. Each virtual patient is characterised by a parameter set sampled from a PBPK/PD population distribution; simulating 5000 virtual patients through 24-week dose schedules requires 5000 independent ODE trajectories, each with ~50 compartments and hundreds of time steps. GPU-parallel batched ODE integration reduces trial simulation time from hours to seconds. Optimal virtual trial design uses GPU-resident Bayesian optimisation over dose/schedule space.
- **Key algorithms:** Monte Carlo virtual population generation, population PBPK/PD ODE integration, Latin hypercube sampling of parameter space, Bayesian optimisation of trial design parameters (dose, schedule, N), survival analysis on simulated endpoints, regulatory-grade power calculation, sensitivity analysis (Morris screening, Sobol indices).
- **Datasets:**
  - Open Systems Pharmacology virtual patient databases (https://github.com/Open-Systems-Pharmacology/)
  - ClinicalTrials.gov schema — trial design parameters for calibration (https://clinicaltrials.gov/)
  - FDA CDER pharmacometric review datasets (verify URL via FDA)
  - Published dose-finding trial datasets in CDISC format (https://www.cdisc.org/)
- **Starter repos/tools:**
  - Pumas (https://pumas.ai/) — GPU-accelerated virtual clinical trials in Julia
  - nvQSP (https://github.com/NVIDIA-Digital-Bio/nvQSP) — GPU PBPK ODE solver for virtual patient simulation
  - SimBiology (MATLAB Parallel Computing Toolbox) — virtual trial simulation with cluster/GPU backend (verify URL)
  - PKPD Simulator (verify URL) — open Python framework for virtual trial simulation
- **CUDA libraries & GPU pattern:** CUDA batched RK45 for thousands of simultaneous patient ODE trajectories, cuRAND for virtual population parameter sampling, Thrust for summary statistic aggregation; pattern: SIMD-parallel ODE integration with each virtual patient in a CUDA warp.

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
