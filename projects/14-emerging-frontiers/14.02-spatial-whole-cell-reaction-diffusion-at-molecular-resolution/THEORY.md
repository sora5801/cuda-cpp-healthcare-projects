# THEORY — 14.2 Spatial / Whole-Cell Reaction-Diffusion at Molecular Resolution

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

### 14.2 Spatial / Whole-Cell Reaction-Diffusion at Molecular Resolution 🔴 · Frontier/Theoretical

- **Deep dive:** Particle-based reaction-diffusion (PBRD) simulators track each molecule as an individual particle, enabling sub-micron spatial resolution of signaling gradients, receptor clustering, and organelle targeting. GPU-accelerated PBRD (Smoldyn GPU, ReaDDy GPU) parallelizes over molecules: each particle diffuses and reacts independently, with nearest-neighbor checks via GPU cell-list algorithms. A full cytoplasm simulation at molecular resolution for even a minimal cell (~500 K unique molecules) at physiologically relevant timescales (milliseconds) requires O(10¹²) timestep-particle updates — tractable only on multi-GPU systems. eGFRD (enhanced Green's Function Reaction Dynamics) is theoretically the most accurate but computationally costly, a prime GPU target.
- **Key algorithms:** Brownian dynamics with reaction (Smoluchowski), eGFRD Green's function propagators, interaction-site model (ISSA), diffusion-limited reaction kernel sampling, GPU cell-list O(N) neighbor search, reactive molecular dynamics.
- **Datasets:** CellOrganizer — generative models of subcellular morphology for simulation domains (http://www.cellorganizer.org/); PDB molecular crowding configurations; SBML-spatial format models (BioModels); MCell neural synapse models (https://mcell.org/).
- **Starter repos/tools:** ReaDDy (https://github.com/readdy/readdy) — GPU-accelerated particle-based RD (CPU + GPU backends); Smoldyn (https://github.com/ssandrews/Smoldyn) — off-lattice GPU-capable PBRD; MCell (https://mcell.org/) — Monte Carlo 3D reaction-diffusion for neurons; STEPS (https://github.com/CNS-OIST/STEPS) — tetrahedral-mesh spatial SSA with GPU support.
- **CUDA libraries & GPU pattern:** CUDA cell-list neighbor search (one thread per particle for neighbor pair collection), cuRAND for per-particle Brownian displacement sampling, Thrust for reaction-event sorting; pattern: GPU cell-list built from particle positions → parallel Brownian displacement → reaction probability check for each particle pair → acceptance-rejection sampling → time step advance.

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
