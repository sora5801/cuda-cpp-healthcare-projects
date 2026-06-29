# THEORY — 13.12 Exposure-Response & Dose-Response Modelling

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

### 13.12 Exposure-Response & Dose-Response Modelling 🟡 · Active R&D

- **Deep dive:** Quantifies the relationship between drug exposure metrics (AUC, Cmax, trough) and clinical or safety endpoints (tumour response, biomarker change, toxicity probability) using GPU-accelerated nonlinear regression and machine learning. In dose-finding trials (Phase I/II), Bayesian model-based dose-escalation designs (EWOC, mTPI-2, BLRM) require rapid posterior sampling after each dose cohort — GPU-accelerated MCMC provides the turnaround speed needed for within-day decision support. Sigmoidal Emax models, logistic regression, and exposure-toxicity models are fitted to cumulative clinical datasets with GPU-parallel gradient computation. The key bottleneck is running thousands of simulated future trial realisations in parallel for adaptive design decision criteria.
- **Key algorithms:** Sigmoidal Emax / Hill equation fitting, Bayesian Logistic Regression Model (BLRM), Escalation With Overdose Control (EWOC), modified Toxicity Probability Interval (mTPI), Emax-time models, power models, direct vs. indirect response PD models, mixture models for responder/non-responder subpopulations, concordance dose-response index.
- **Datasets:**
  - FDA Pharmacometrics Reviews — dose-response data from NDA/BLA submissions (https://www.fda.gov/drugs/drug-approvals-and-databases/pharmacometrics-reviews)
  - Published dose-escalation trial data in Oncology (verify individual publications)
  - DoseFinding R package example datasets (verify URL)
  - CDISC ADaM dose-response trial data formats (https://www.cdisc.org/)
- **Starter repos/tools:**
  - Pumas (https://pumas.ai/) — GPU Bayesian E-R modelling in Julia
  - DoseFinding R package (https://cran.r-project.org/web/packages/DoseFinding/) — classical dose-finding model fitting
  - BOIN (https://cran.r-project.org/web/packages/BOIN/) — Bayesian optimal interval design for dose-finding
  - trialDesign (verify URL) — simulation platform for adaptive dose-escalation
- **CUDA libraries & GPU pattern:** cuRAND for Monte Carlo posterior simulation, cuBLAS for sigmoidal Emax regression Hessians, custom CUDA kernels for parallel trial simulation over candidate dose levels; pattern: GPU-parallel simulation of thousands of adaptive trial scenarios for decision support.

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
