# THEORY — 10.13 3D Bioprinting Toolpath & Bioink Process Simulation

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

### 10.13 3D Bioprinting Toolpath & Bioink Process Simulation 🟡 · Active R&D

- **Deep dive:** Extrusion-based bioprinting deposits cell-laden hydrogels through a nozzle, where shear stress during extrusion determines post-print cell viability. GPU-accelerated CFD of the nozzle + deposition region (non-Newtonian Carreau fluid) predicts wall shear stress as a function of nozzle geometry, ink rheology, and print speed, enabling parameter optimization in silico before costly biological experiments. Lattice-structure scaffold design — maximizing permeability for nutrient transport while maintaining mechanical stiffness — uses GPU topology optimization with fluid-flow homogenization. Thermal modeling of photopolymerization in DLP/SLA bioprinting on GPU resolves crosslink-front propagation in real time.
- **Key algorithms:** Non-Newtonian Navier-Stokes (Carreau-Yasuda viscosity model), topology optimization with permeability (Darcy-Stokes coupling), heat-transfer / photo-crosslinking kinetics, support-structure generation via GPU ray casting, ML surrogate (XGBoost/MLP) for viability prediction.
- **Datasets:** In silico Bioink Viability Dataset (Zenodo) — extrusion viability vs. shear-stress features (https://zenodo.org/records/11545357); BioInk Rheology Database (verify URL via Biofabrication journal); 3D Bioprinting Benchmarks (verify URL via Zenodo); Scaffold Permeability Benchmark (https://arxiv.org/abs/1104.1028).
- **Starter repos/tools:** in-silico-bioink-viability-prediction (https://github.com/KORINZ/in-silico-bioink-viability-prediction) — ML viability prediction from shear stress; OpenFOAM (https://github.com/OpenFOAM) — non-Newtonian flow solver for nozzle CFD; FEBio (https://github.com/febiosoftware/FEBio) — scaffold mechanical FEA; TPMS Scaffold Generator (verify URL via GitHub) — GPU-accelerated triply-periodic-minimal-surface lattice generation.
- **CUDA libraries & GPU pattern:** CUDA kernels for non-Newtonian viscosity update per cell, cuFFT for spectral pressure solve, cuDNN for surrogate viability model inference; pattern: parametric nozzle geometry → GPU Navier-Stokes solve for shear-stress field → shear-stress statistics fed to GPU ML surrogate → output: print parameters vs. predicted viability Pareto front.

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
