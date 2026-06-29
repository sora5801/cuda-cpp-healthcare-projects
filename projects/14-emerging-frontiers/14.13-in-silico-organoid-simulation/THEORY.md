# THEORY — 14.13 In Silico Organoid Simulation

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

### 14.13 In Silico Organoid Simulation 🔴 · Frontier/Theoretical

- **Deep dive:** Organoids — self-organizing 3D stem-cell-derived mini-organs — grow via coupled cell division, differentiation, migration, and mechanical deformation. GPU-accelerated vertex models, cellular Potts models (CPM), and off-lattice agent-based models (ABMs) simulate organoid morphogenesis across thousands to millions of cells. A key bottleneck is computing cell-cell contact forces and sorting energies for CPM (Metropolis Monte Carlo), which are embarrassingly parallel over lattice sites. Virtual tissue simulation from real image data (Frontiers 2024) uses GPU-segmented confocal images to initialize physics-based organoid models, enabling patient-specific drug response prediction for personalized oncology.
- **Key algorithms:** Cellular Potts Model (CPM) Metropolis Monte Carlo, vertex model for epithelial mechanics, off-lattice center-based model (CBM), reaction-diffusion morphogen fields (Turing), subcellular element model (SEM), mechanical feedback on gene regulatory network.
- **Datasets:** Kaggle Sartorius Cell Instance Segmentation (https://www.kaggle.com/c/sartorius-cell-instance-segmentation); OpenCell — protein localization in live cells (https://opencell.czbiohub.org/); CancerOrganoidDB — organoid drug response (verify URL via Hubrecht Institute); NeurIPS Cell Seg Challenge organoid images (verify URL via Grand Challenge).
- **Starter repos/tools:** CompuCell3D (https://compucell3d.org/) — GPU-capable CPM organoid simulation; Morpheus (https://morpheus.gitlab.io/) — GPU cellular Potts + reaction-diffusion; Chaste (https://github.com/Chaste/Chaste) — off-lattice ABM for organoid growth; PhysiCell (https://github.com/MathCancer/PhysiCell) — 3D agent-based multicellular GPU-parallelized simulator.
- **CUDA libraries & GPU pattern:** CUDA checkerboard-parallel Metropolis updates for CPM (even/odd lattice coloring), CUDA reaction-diffusion 3D stencils, cuRAND for Monte Carlo move proposals; pattern: organoid image segmentation → GPU initialization of CPM lattice → parallel Metropolis sweeps (checkerboard coloring avoids conflicts) → reaction-diffusion morphogen update → cell-fate decision → geometry output for imaging comparison.

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
