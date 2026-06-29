# THEORY — 9.7 Vaccine Allocation & Intervention Optimisation

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

### 9.7 Vaccine Allocation & Intervention Optimisation 🟡 · Active R&D

- **Deep dive:** Determines optimal allocation of limited vaccines, treatments, or non-pharmaceutical interventions across age groups, geographic regions, or risk strata to minimise deaths or infections under resource constraints. GPU-accelerated simulation (agent-based or compartmental) enables rapid evaluation of thousands of candidate allocation policies within an optimisation loop. Reinforcement learning approaches (PPO, SAC) train on GPU-simulated environments where the epidemic simulator is the transition function. Multi-objective Pareto optimisation across equity and efficiency criteria requires GPU-parallelised NSGA-II or similar evolutionary algorithms.
- **Key algorithms:** Multi-objective optimisation (NSGA-II, NSGA-III), Proximal Policy Optimisation (PPO), Deep Q-Networks on simulation environments, Thompson sampling for adaptive allocation, network-based vaccinating-hub strategies (targeted vs. random), stochastic programming under epidemiological uncertainty, integer linear programming for logistics.
- **Datasets:**
  - GLEAM global mobility network for spatial allocation (https://www.gleamviz.org/)
  - WHO Immunisation Data — vaccination coverage by country and vaccine (https://immunizationdata.who.int/)
  - US Census commuting flows — for workplace transmission modelling (https://www.census.gov/)
  - COVID-19 vaccination time series (Our World in Data) — historical rollout data for calibration (https://ourworldindata.org/covid-vaccinations)
- **Starter repos/tools:**
  - Covasim (https://github.com/InstituteforDiseaseModeling/covasim) — GPU-friendly Python COVID-19 agent-based model
  - EMOD (https://github.com/InstituteforDiseaseModeling/EMOD) — high-performance individual-based disease model
  - Stable Baselines 3 (https://github.com/DLR-RM/stable-baselines3) — GPU RL library for policy training on epidemic environments
  - Pymoo (https://github.com/anyoptimization/pymoo) — multi-objective optimisation with GPU evaluation support
- **CUDA libraries & GPU pattern:** cuRAND for stochastic epidemic simulation, custom CUDA ODE kernels for compartmental model evaluation, CUDA graph for repeated fixed-topology GPU execution; pattern: population of candidate policies evaluated simultaneously across GPU thread blocks.

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
