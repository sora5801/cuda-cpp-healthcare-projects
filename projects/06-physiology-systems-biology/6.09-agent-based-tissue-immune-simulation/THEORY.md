# THEORY — 6.9 Agent-Based Tissue / Immune Simulation

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

### 6.9 Agent-Based Tissue / Immune Simulation 🟡 · Active R&D
- **Deep dive:** Tissue is modeled as a population of autonomous agents (cells) each tracking position, velocity, cycle state, secretion rates, and mechanistic signaling. Cell-cell mechanical interactions (overlap repulsion, adhesion) require pairwise neighbor search that scales as O(N²) naively but drops to O(N) with spatial binning on GPU. Immune cell migration, cytokine diffusion, and tumor-immune coevolution are natural applications. PhysiCell supports 10⁵–10⁶ cells in 3D with GPU-accelerated substrate diffusion.
- **Key algorithms:** Center-based mechanics (soft-sphere repulsion + adhesion), cell cycle models (Ki67 basic/advanced, flow cytometry), substrate diffusion (Thomas ADI or explicit FD on Cartesian grid), chemotaxis gradient following, receptor-ligand binding kinetics, Boolean intracellular signaling (MaBoSS), spatial hashing for neighbor search.
- **Datasets:** CancerSEA single-cell functional states (http://biocc.hrbmu.edu.cn/CancerSEA/); TCGA pan-cancer immune landscape (https://portal.gdc.cancer.gov); MIBI/IMC imaging mass cytometry datasets (various Zenodo deposits); TCIA immunotherapy imaging (https://www.cancerimagingarchive.net).
- **Starter repos/tools:** PhysiCell (https://github.com/MathCancer/PhysiCell) — 3D multicellular simulator with physics + biotransport; PhysiBoSS (https://github.com/PhysiBoSS/PhysiBoSS) — Boolean network–PhysiCell coupling for signaling; Chaste (https://github.com/Chaste/Chaste) — off-lattice cell-based models with vertex/Voronoi mechanics; MOOSE (https://github.com/BhallaLab/moose-core) — chemical signaling within cells.
- **CUDA libraries & GPU pattern:** Custom CUDA kernels for substrate PDE (explicit or ADI); CUDA Thrust for cell sort by spatial bin; atomic-add for cytokine source terms from agent loop; pattern: hybrid CPU (agent logic) + GPU (PDE + neighbor search) with pinned memory for cell state transfer.

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
