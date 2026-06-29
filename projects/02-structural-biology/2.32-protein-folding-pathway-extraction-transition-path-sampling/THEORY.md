# THEORY — 2.32 Protein Folding Pathway Extraction (Transition Path Sampling)

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

### 2.32 Protein Folding Pathway Extraction (Transition Path Sampling) 🔴 · Frontier/Theoretical

- **Deep dive:** Transition Path Sampling (TPS) harvests rare folding/unfolding events by shooting from configurations near the transition state and accepting/rejecting trajectories that connect folded and unfolded basins. GPU MD makes it practical to run many short (~1–100 ns) shooting moves in parallel. AIMMD (AI-augmented MD) uses GPU-trained neural networks to identify committor isosurfaces, accelerating TPS convergence. Applications include protein folding mechanism elucidation, cryptic pocket opening pathways, and drug unbinding kinetics (τRAMD, WExplore).
- **Key algorithms:** Transition path sampling (TPS) shooting move, aimless shooting, committor analysis, path collective variables (PathCV), weighted ensemble sampling (WExplore/WE-H), τRAMD unbinding kinetics, AIMMD neural committor.
- **Datasets:** Anton/Shaw millisecond trajectories as TPS starting configurations; GPCRmd pathway datasets (https://gpcrmd.org); folding benchmarks: Trp-cage, chignolin, WW domain; SAMPL host-guest kinetics challenges (verify URL).
- **Starter repos/tools:** OpenPathSampling (https://github.com/openpathsampling/openpathsampling) — TPS on GPU via OpenMM; HTMD (https://github.com/Acellera/htmd) — GPU-accelerated adaptive sampling; WESTPA (https://westpa.github.io/westpa/) — weighted ensemble sampling on GPU MD; AIMMD (https://github.com/bioRxiv AIMMD, verify URL) — AI-augmented TPS with GPU neural committor.
- **CUDA libraries & GPU pattern:** GPU MD for fast shooting trajectories; GPU neural network committor inference in AIMMD; NCCL for WE parent-child trajectory coordination; embarrassingly parallel independent shooter array on multi-GPU.

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
