# THEORY — 9.3 Contact-Network & Graph Epidemic Dynamics

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

### 9.3 Contact-Network & Graph Epidemic Dynamics 🟡 · Active R&D

- **Deep dive:** Simulates epidemic spread on empirical or synthetic contact networks where nodes are individuals and weighted edges encode contact intensity. GPU graph traversal (BFS/DFS) across networks with millions of nodes enables exploration of counterfactual intervention scenarios (edge removal, node vaccination) in seconds vs. hours on CPU. The Replay tool transforms empirical timestamped contact data into duration-weighted adjacency matrices and uses GPU sparse matrix operations for realistic epidemic simulation. cuGraph's PageRank and community detection accelerate identification of superspreader hubs for targeted interventions.
- **Key algorithms:** SIR/SEIR stochastic simulation on contact graphs, Gillespie algorithm for continuous-time Markov chains, non-Markovian renewal kernels (FlashSpread), Belief Propagation for marginal inference on sparse graphs, community detection (Louvain, Leiden), targeted vaccination on high-degree nodes, R0 spectral radius estimation.
- **Datasets:**
  - SocioPatterns proximity contact data — face-to-face contacts in hospitals, schools, conferences (http://www.sociopatterns.org/)
  - Copenhagen Networks Study — Bluetooth proximity + mobile data for 800 students (verify URL)
  - GLEAM global mobility network (https://www.gleamviz.org/)
  - NiemaGraphGen synthetic contact networks (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10038133/) — memory-efficient global-scale simulation toolkit
- **Starter repos/tools:**
  - FlashSpread (https://arxiv.org/abs/2604.22092) — GPU framework for network epidemic dynamics (verify GitHub URL)
  - Replay (https://link.springer.com/article/10.1186/s12911-025-03310-2) — GPU-accelerated temporal contact network epidemiology tool
  - cuGraph (https://github.com/rapidsai/cugraph) — GPU graph analytics (PageRank, BFS, community detection) via RAPIDS
  - EoN (Epidemics on Networks) (https://github.com/springer-math/Mathematics-of-Epidemics-on-Networks) — Python network epidemic simulation
- **CUDA libraries & GPU pattern:** cuGraph BFS/SSSP for infection spread on GPU-resident adjacency, cuSPARSE SpMV for transition probability matrices, cuRAND for stochastic edge activation; pattern: BFS-based wavefront parallelism with atomic state update per node.

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
