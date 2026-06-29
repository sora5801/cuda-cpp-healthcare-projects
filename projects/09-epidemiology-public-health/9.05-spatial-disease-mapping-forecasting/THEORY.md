# THEORY — 9.5 Spatial Disease Mapping & Forecasting

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

### 9.5 Spatial Disease Mapping & Forecasting 🟡 · Active R&D

- **Deep dive:** Estimates disease incidence surfaces and spatiotemporal risk across geographic grids using Bayesian geostatistical models (BYM, INLA, Gaussian Process regression). The Gaussian process kernel matrix computation scales as O(N²) in the number of spatial locations — for a 10k-pixel grid this is a 10⁸-element covariance matrix, whose Cholesky decomposition is dominated by GPU-accelerated dense linear algebra (cuBLAS). GPU-based MCMC samplers (BlackJAX on CUDA, Greta with GPU backend) achieve 380× speedup for epidemic forecasting models. Interpolating national case-counts to sub-district resolution using kriging is entirely parallelisable across prediction locations on GPU.
- **Key algorithms:** Besag-York-Mollié (BYM) spatial smoothing, Integrated Nested Laplace Approximation (INLA), Gaussian Process regression, kriging interpolation, spatiotemporal Kalman filtering, Bayesian hierarchical Poisson regression, neural ODE spatial models, ensemble Kalman filters.
- **Datasets:**
  - WHO Mortality Database — ICD-coded deaths by country and cause (https://www.who.int/data/data-collection-tools/who-mortality-database)
  - IHME Global Burden of Disease — country-level disease incidence estimates (https://www.healthdata.org/gbd)
  - CDC Wonder — US county-level disease surveillance data (https://wonder.cdc.gov/)
  - NASA SEDAC Global Population Data — gridded population for exposure modelling (https://sedac.ciesin.columbia.edu/)
- **Starter repos/tools:**
  - INLA / R-INLA (https://www.r-inla.org/) — fast Bayesian spatial modelling; GPU via PARDISO sparse solver
  - BlackJAX (https://github.com/blackjax-devs/blackjax) — GPU-accelerated Bayesian sampling (HMC, NUTS) via JAX
  - Greta (https://github.com/greta-dev/greta) — probabilistic programming with TensorFlow GPU backend for spatial models
  - CARBayes (https://github.com/duncanplee/CARBayes) — R package for spatial Bayesian modelling (CPU; parallelisable)
- **CUDA libraries & GPU pattern:** cuBLAS for GP covariance matrix Cholesky, cuSPARSE for ICAR precision matrix operations, JAX XLA for GPU-accelerated MCMC; pattern: batch kriging over prediction grid points with fully GPU-resident covariance kernel.

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
