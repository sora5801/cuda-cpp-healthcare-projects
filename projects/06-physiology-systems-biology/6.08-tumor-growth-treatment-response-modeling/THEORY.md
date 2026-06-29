# THEORY — 6.8 Tumor Growth & Treatment-Response Modeling

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

### 6.8 Tumor Growth & Treatment-Response Modeling 🟡 · Active R&D
- **Deep dive:** Continuum-PDE models (reaction-diffusion for nutrient/oxygen, tumor cell density, and treatment drug concentration) combined with discrete cell-based models capture avascular-to-vascular tumor growth, hypoxia-driven necrosis, and response to radiation or chemotherapy. GPU acceleration is essential for solving coupled PDE systems on 3D grids (512³ voxels = 1.3×10⁸ cells) at each time step of a multi-day simulation. Parameter sweeps for virtual clinical trials (thousands of parameter sets) are embarrassingly parallel across the GPU grid.
- **Key algorithms:** Fisher-KPP reaction-diffusion (tumor cell density), oxygen/nutrient diffusion-consumption (Green's function or FD), phenomenological radiobiological model (linear-quadratic), drug PK/PD compartment coupling, phase-field tumor morphology, level-set interface tracking for tumor boundary.
- **Datasets:** TCGA (The Cancer Genome Atlas) — multi-omics + imaging for model calibration (https://portal.gdc.cancer.gov); TCIA (The Cancer Imaging Archive) — multi-institutional tumor imaging (https://www.cancerimagingarchive.net); PhysioNet oncology waveforms (https://physionet.org); Zenodo tumor growth simulation datasets (search zenodo.org for "tumor growth simulation").
- **Starter repos/tools:** PhysiCell (https://github.com/MathCancer/PhysiCell) — 3D agent-based multicellular simulator with diffusing substrates, scales linearly in cell count; PhysiBoSS (https://github.com/PhysiBoSS/PhysiBoSS) — extends PhysiCell with Boolean network intracellular signaling (MaBoSS); Chaste (https://github.com/Chaste/Chaste) — includes tumor spheroid and crypt models; OpenFOAM (https://github.com/OpenFOAM/OpenFOAM-dev) — used for drug delivery flow simulations.
- **CUDA libraries & GPU pattern:** Custom CUDA FD stencil kernels (3D 7-point Laplacian on oxygen/drug grids), CUDA Thrust for per-cell agent sorting and binning, cuRAND for stochastic division/death events; pattern: 3D CUDA thread grid for PDE, separate kernel for agent-based cell loop with shared-memory neighborhood queries.

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
