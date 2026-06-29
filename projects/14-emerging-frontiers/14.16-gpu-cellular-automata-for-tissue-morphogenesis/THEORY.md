# THEORY — 14.16 GPU Cellular Automata for Tissue Morphogenesis

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

### 14.16 GPU Cellular Automata for Tissue Morphogenesis 🟡 · Active R&D

- **Deep dive:** Lattice-Gas Cellular Automata (LGCA) and Cellular Automata (CA) models simulate tumor invasion, wound healing, and developmental tissue patterning at the cell scale on million-element grids. Every lattice site updates in parallel based on local neighborhood rules — a perfectly SIMT workload. GPU CA for tumor growth integrates nutrient diffusion (CUDA stencil), cell-cycle progression, and proliferation/death rules, enabling parameter sweeps over invasion phenotypes that would be intractable on CPU. Hybrid CA-PDE models couple discrete cell lattice (CUDA) with continuous nutrient/oxygen fields (CUDA finite difference).
- **Key algorithms:** Lattice-Gas CA (LGCA) for cell migration, Cellular Automaton tumor model (Kansal-Torquato), Go-or-Grow phenotype switching, reaction-diffusion PDE for morphogens, Potts model for cell sorting, hybrid CA-FEM multiscale coupling.
- **Datasets:** CancerOrganoid Drug Response Images (verify URL via Hubrecht); TCGA pathology slides for CA calibration (https://portal.gdc.cancer.gov/); CellMorph — time-lapse cell migration datasets (verify URL); Wound-Healing Assay Image Repository (verify URL via protocols.io).
- **Starter repos/tools:** PhysiCell (https://github.com/MathCancer/PhysiCell) — GPU-parallelized 3D agent-based tissue simulator; CompuCell3D (https://compucell3d.org/) — multi-algorithm tissue simulator with GPU support; CancerSim (https://github.com/joancalvente/cancersim) — GPU CA tumor growth code; Morpheus (https://morpheus.gitlab.io/) — spatial cell model simulation with GPU backend.
- **CUDA libraries & GPU pattern:** CUDA 2D/3D stencil kernels for CA lattice update, cuRAND for stochastic cell-fate decisions, Thrust for parallel phenotype census; pattern: N×N×N GPU lattice → one CUDA thread per lattice site → local rule evaluation → stochastic update → reaction-diffusion field update → time-step advance → GPU-rendered morphology export.

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
