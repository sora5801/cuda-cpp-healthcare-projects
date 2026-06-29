# THEORY — 13.11 Bayesian PK/PD Inference on GPU (Torsten/Stan)

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

### 13.11 Bayesian PK/PD Inference on GPU (Torsten/Stan) 🟡 · Active R&D

- **Deep dive:** Full Bayesian inference for PK/PD models using Hamiltonian Monte Carlo (HMC/NUTS) within Stan + Torsten, where the log-posterior gradient requires integrating population ODE trajectories and evaluating the likelihood. Each HMC leapfrog step requires one full ODE solve per patient in the dataset — for 1000 patients × 2000 HMC iterations × 10 leapfrog steps = 20M ODE solves per chain. GPU acceleration of these batched ODE solves provides the critical speedup. The `reduce_sum` function in Stan enables within-chain parallelism across patients on multi-core CPU; true GPU acceleration requires the CUDA ODE integration backends available through Pumas or experimental Stan GPU interfaces.
- **Key algorithms:** Hamiltonian Monte Carlo (HMC), No-U-Turn Sampler (NUTS), automatic differentiation through ODE solvers (adjoint sensitivity), Runge-Kutta ODE integration, adaptive dual-averaging stepsize, Bayesian predictive check (PPC), R-hat convergence diagnostics, Bayesian cross-validation (LOO-CV).
- **Datasets:**
  - Torsten example models (https://github.com/metrumresearchgroup/Torsten) — 2-compartment, PKPD, TMDD Stan models
  - Somatrogon population PK dataset (ResearchGate, 2024) — Bayesian NLME application with Torsten
  - Warfarin PK/PD dataset — standard Bayesian NLME benchmark (verify URL)
  - MIMIC-IV medication + lab values — vancomycin TDM for Bayesian dosing (https://physionet.org/content/mimiciv/)
- **Starter repos/tools:**
  - Torsten (https://github.com/metrumresearchgroup/Torsten) — Stan ODE extensions for PK/PD; SAEM and HMC
  - CmdStanR / CmdStanPy (https://mc-stan.org/cmdstanr/) — Stan interface for running GPU-parallel chains
  - Pumas (https://pumas.ai/) — Julia Bayesian PK/PD with GPU-accelerated HMC via CUDA.jl
  - MCMCChains (https://github.com/TuringLang/MCMCChains.jl) — MCMC diagnostics for population PK posteriors
- **CUDA libraries & GPU pattern:** GPU-parallelised ODE solvers called from Stan adjoint sensitivity method, cuBLAS for Hessian approximation, NCCL for multi-chain parallelism; pattern: multi-GPU chains run in parallel with NCCL synchronisation for diagnostics.

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
