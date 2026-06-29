# THEORY — 9.2 Large-Scale Compartmental & Metapopulation Models

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

### 9.2 Large-Scale Compartmental & Metapopulation Models 🟡 · Active R&D

- **Deep dive:** Solves large systems of ODEs or stochastic differential equations (SDEs) describing disease dynamics across thousands of geographic patches interconnected by mobility flows (SIR at metapopulation scale, seasonal forcing, age structure). ODE integration over thousands of patches with coupling matrices is equivalent to a batched sparse matrix-vector multiply at each time step — a cuSPARSE-accelerated operation. Monte Carlo uncertainty quantification requires thousands of independent ODE solves in parallel on GPU, each with different parameter samples. GPU-based adaptive stepsize RK4/5 solvers (Torchdiffeq's `dopri5` on GPU) handle stiff biological dynamics efficiently.
- **Key algorithms:** Runge-Kutta 4/5 ODE integration on GPU, tau-leaping for stochastic compartmental models, MCMC parameter inference (ensemble MCMC), Approximate Bayesian Computation (ABC), metapopulation coupling via mobility matrix, seasonal forcing with Fourier series, age-structured SEIR with contact matrices.
- **Datasets:**
  - GLEAM — global airline + commuting network for metapopulation coupling (https://www.gleamviz.org/)
  - WHO Weekly Epidemiological Reports — case counts for parameter calibration (https://www.who.int/emergencies/situations)
  - CDC FluView — US influenza surveillance by week and region (https://www.cdc.gov/flu/weekly/)
  - COVID-19 Data Repository by CSSE at Johns Hopkins (archived) — global case/death time series (https://github.com/CSSEGISandData/COVID-19)
- **Starter repos/tools:**
  - Epiflows / EpiModel (https://github.com/EpiModel/EpiModel) — network-based compartmental modelling in R
  - Torchdiffeq (https://github.com/rtqichen/torchdiffeq) — GPU-accelerated neural ODE and standard ODE solvers
  - MEmilio (https://github.com/SciCompMod/memilio) — high-performance C++/CUDA epidemic simulation
  - PyGOM (https://github.com/ukhsa-collaboration/pygom) — Python compartmental ODE modelling framework
- **CUDA libraries & GPU pattern:** cuSPARSE for mobility matrix coupling, cuRAND for stochastic tau-leaping, custom RK4 CUDA kernel for parallel ODE batch; pattern: each CUDA thread block integrates one metapopulation patch ODE system, with shared memory holding coupling matrices.

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
