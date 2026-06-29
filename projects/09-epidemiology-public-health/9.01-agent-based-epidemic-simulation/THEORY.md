# THEORY — 9.1 Agent-Based Epidemic Simulation

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

### 9.1 Agent-Based Epidemic Simulation 🟡 · Active R&D

- **Deep dive:** Simulates individual-level epidemic spread across millions of synthetic agents, each with behavioural rules governing contact, infection, and recovery. GPU parallelism maps each agent to a thread or thread group: state updates (susceptible → exposed → infectious → recovered) are embarrassingly parallel across the population. The bottleneck is computing pairwise contacts within spatial proximity grids or synthetic social networks; cuGraph adjacency traversal accelerates this. Non-Markovian (renewal) dynamics require tracking each agent's infectious age distribution, a memory-intensive operation that fits within GPU SRAM when using compressed state representations. FlashSpread achieves end-to-end GPU execution with kernel-fused dense stepping.
- **Key algorithms:** SIR/SEIR/SEIRD state machines per agent, contact kernel simulation (household, workplace, school stratification), non-Markovian renewal spreading, GPU-parallel BFS over contact graphs, Monte Carlo ensemble averaging, importance sampling for rare events, spatial hashing for local contact discovery.
- **Datasets:**
  - GLEAM / GLEaMviz global mobility + population data (https://www.gleamviz.org/)
  - US Census TIGER/Line shapefiles + ACS commuting data (https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html)
  - Mossong et al. POLYMOD contact matrices — age-structured contact rates across 8 European countries (verify URL)
  - SafeGraph / Dewey mobility data — retail foot traffic and mobility patterns (verify URL)
- **Starter repos/tools:**
  - FRED (Framework for Reconstructing Epidemic Dynamics) (https://github.com/PublicHealthDynamicsLab/FRED) — individual-level US epidemic simulator
  - FlashSpread (https://arxiv.org/abs/2604.22092) — end-to-end GPU framework for non-Markovian network spreading (verify GitHub URL)
  - MEmilio (https://github.com/SciCompMod/memilio) — high-performance modular epidemic simulation software with GPU support
  - Epiabm (https://github.com/RESIDE-ICL/epiabm) — GPU-parallelised ABM framework for epidemic simulation
- **CUDA libraries & GPU pattern:** cuGraph for contact network BFS/DFS, cuRAND for stochastic transition sampling, custom CUDA kernels for per-agent state update; pattern: one CUDA thread per agent with shared-memory contact lookup tables, warp-level primitives for neighbour enumeration.

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
