# THEORY — 11.5 Bioreactor & Fermentation CFD

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

### 11.5 Bioreactor & Fermentation CFD 🟡 · Active R&D

- **Deep dive:** Industrial bioreactors exhibit complex turbulent flow, gas-liquid mass transfer (O₂/CO₂), and biological reactions that mutually couple over timescales from milliseconds (bubble coalescence) to hours (cell growth). GPU-accelerated LBM or finite-volume CFD resolves the multi-phase (broth + bubbles) hydrodynamics on meshes with millions of cells, enabling scale-up prediction from bench to 10,000-L fermenters. CFD-metabolic hybrid models link local glucose/O₂ concentrations (from CFD) to spatially-resolved metabolic rates (from flux-balance analysis), identifying gradients that stress industrial cultures. Real-time digital twins combining online sensor data with GPU CFD surrogates enable closed-loop bioreactor control.
- **Key algorithms:** Turbulent Navier-Stokes (k-ε / k-ω SST), volume-of-fluid (VOF) gas-liquid interface, population balance model for bubble size distribution, Euler-Euler two-phase flow, lattice-Boltzmann for pore-scale mass transfer, physics-informed neural network surrogate, computational morphology (impeller blade design).
- **Datasets:** DECHEMA Bioreactor Flow Dataset — PIV measurements in stirred tanks (verify URL via dechema.de); OpenFOAM BioReactor Tutorial Cases (https://www.openfoam.com/); CHO Fed-Batch Time Course Data (BioNumbers DB, https://bionumbers.hms.harvard.edu/); Zenodo fermentation monitoring datasets (search Zenodo "fed-batch bioreactor").
- **Starter repos/tools:** OpenFOAM (https://github.com/OpenFOAM) — gas-liquid bioreactor multiphase solvers (multiphaseEulerFoam) with GPU linear algebra; Palabos (https://gitlab.com/unigespc/palabos) — GPU LBM for porous-media and bubble-column flows; NVIDIA PhysicsNeMo (https://github.com/NVIDIA/physicsnemo) — physics-informed surrogate training for CFD; COBRApy (https://github.com/opencobra/cobrapy) — flux-balance metabolic modeling for CFD coupling.
- **CUDA libraries & GPU pattern:** cuSPARSE for pressure-velocity coupling in SIMPLE algorithm, CUDA kernels for VOF interface reconstruction, cuDNN for PINN surrogate inference; pattern: full CFD on GPU with AMG preconditioner → extract local O₂/glucose fields → pass to GPU flux-balance metabolic model → update volumetric reaction terms → iterate time step.

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
