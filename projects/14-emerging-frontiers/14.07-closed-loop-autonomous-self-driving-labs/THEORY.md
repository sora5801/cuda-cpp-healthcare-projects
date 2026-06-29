# THEORY — 14.7 Closed-Loop Autonomous "Self-Driving" Labs

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

### 14.7 Closed-Loop Autonomous "Self-Driving" Labs 🔴 · Frontier/Theoretical

- **Deep dive:** Self-driving labs (SDLs) close the design-build-test-learn cycle by coupling GPU-accelerated Bayesian optimization (BO) or reinforcement learning to robotic liquid handlers, automated assays, and real-time data pipelines. The GPU role is the inner-loop inference: scoring thousands of candidate experiments via surrogate models (GP, neural network ensembles) in milliseconds, so the acquisition function evaluates faster than the robot can dispense. Active learning for drug discovery (e.g., Gaussian Process + batch BO with qEI) has been shown to find optima in 10–50× fewer experiments. Photonic lab automation systems integrate GPU-accelerated spectroscopic analysis (Raman, fluorescence) for real-time compound characterization.
- **Key algorithms:** Bayesian optimization with Gaussian process (GP-UCB, qEI), neural network ensemble surrogate, multi-fidelity BO, reinforcement learning (PPO for experiment selection), active learning, parallel batch BO (TurBO), uncertainty quantification via deep ensembles or MC dropout.
- **Datasets:** ChEMBL HTS screening data (https://www.ebi.ac.uk/chembl/); Open Reaction Database (ORD) — chemical reaction outcomes (https://open-reaction-database.org/); Therapeutic Data Commons (TDC) — multi-property drug benchmarks (https://tdcommons.ai/); Syngas Fermentation Simulator multi-fidelity dataset (https://arxiv.org/abs/2311.05776).
- **Starter repos/tools:** BoTorch (https://github.com/pytorch/botorch) — GPU Bayesian optimization with PyTorch; Ax (https://github.com/facebook/Ax) — adaptive experimentation platform using BoTorch; Summit (https://github.com/sustainable-processes/summit) — BO library for chemical process optimization; Olympus (https://github.com/aspuru-guzik-group/olympus) — benchmark framework for self-driving lab algorithms.
- **CUDA libraries & GPU pattern:** cuDNN for deep ensemble surrogate inference, Cholesky factorization via cuSolver for GP posterior, GPU-accelerated acquisition function optimization (batch gradient ascent); pattern: prior experiment observations → GPU GP/neural surrogate fit → parallel acquisition function maximization (256 candidates) → top-k experiments dispatched to robot → new measurements update surrogate.

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
