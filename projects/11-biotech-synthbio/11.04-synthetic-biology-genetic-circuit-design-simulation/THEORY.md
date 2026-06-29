# THEORY — 11.4 Synthetic-Biology Genetic-Circuit Design & Simulation

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

### 11.4 Synthetic-Biology Genetic-Circuit Design & Simulation 🟡 · Active R&D

- **Deep dive:** Genetic-circuit design requires stochastic simulation (Gillespie SSA) of regulatory networks with hundreds of species and reactions, then optimization of promoter strengths, RBS sequences, and protein copy numbers to achieve target transfer-function shapes. GPU parallelism runs thousands of independent SSA trajectories simultaneously on a single card — each trajectory is a separate CUDA stream — reducing Monte Carlo ensemble variance estimation from hours to seconds. Deterministic ODE simulation (Hill kinetics) of large gene regulatory networks (GRNs) further benefits from GPU batch-ODE solvers (cuSolver + custom RK4). Bayesian optimization over the genetic parameter space closes the design loop.
- **Key algorithms:** Gillespie Stochastic Simulation Algorithm (SSA), tau-leaping (accelerated SSA), deterministic ODE integration with Hill-function kinetics, Bayesian optimization (GP-UCB) for parameter tuning, coarse-grained thermodynamic models for promoter strength, Boolean logic gate composition.
- **Datasets:** iGEM Registry of Standard Biological Parts — promoter/RBS/gene part catalog (https://parts.igem.org/); SBOL Designer parts library (https://sboldesigner.github.io/); BioBrick Characterization Database (verify URL via SynBioHub); Promoter Strength Library (Anderson promoter series) (verify URL via parts.igem.org).
- **Starter repos/tools:** Tellurium (https://github.com/sys-bio/tellurium) — Python ODE/SSA simulator for SBML models with CUDA-extensible solvers; GillesPy2 (https://github.com/StochSS/GillesPy2) — Python SSA with GPU acceleration roadmap; COPASI (https://github.com/copasi/COPASI) — biochemical network simulator with parallel parameter scanning; iBioSim (https://github.com/MyersResearchGroup/iBioSim) — genetic circuit design + simulation framework.
- **CUDA libraries & GPU pattern:** CUDA kernels for parallel SSA trajectories (one trajectory per thread block), cuRAND for per-trajectory random number streams, cuSolver for stiff ODE Jacobian factorization; pattern: genetic circuit model → 10⁴ GPU SSA trajectories in parallel → histogram-based transfer-function estimation → Bayesian optimizer proposes new promoter parameters → iterate.

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
