# THEORY — 6.5 Respiratory / Lung Airflow & Particle Deposition

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

### 6.5 Respiratory / Lung Airflow & Particle Deposition 🟡 · Active R&D
- **Deep dive:** Simulates inspiratory/expiratory flow through the conducting airways (generations 0–16, reconstructed from CT) and tracks inhaled aerosol/drug particle trajectories via Lagrangian particle tracking. The lung's tree topology means ~10⁶–10⁷ computational cells in the airway geometry and millions of particle trajectories evaluated each breath cycle—both trivially parallelizable on GPU. Alveolar gas exchange adds a reaction-diffusion layer for O₂/CO₂ that couples to a 1D ventilation model for the respiratory tree periphery.
- **Key algorithms:** Incompressible Navier-Stokes (finite volume), Lagrangian discrete-phase particle tracking (drag + Brownian + Saffman lift forces), Stokes drag law, k-ω SST RANS turbulence, LBM for alveolar-scale flow, convection-diffusion for gas species, quasi-1D ventilation model (Horsfield tree).
- **Datasets:** LIDC-IDRI lung CT — 1 010 cases with nodule annotations, TCIA (https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI); COPDGene lung CT dataset — 10 000 subjects (https://www.copdgene.org); SPIROMICS bronchial CT (https://www.spiromics.org); PhysioNet respiratory waveform databases (https://physionet.org).
- **Starter repos/tools:** OpenFOAM-dev (https://github.com/OpenFOAM/OpenFOAM-dev) — Lagrangian particle tracking (DPMFoam) with GPU-capable solver via GPU-accelerated AmgX pressure solve; SimVascular (https://github.com/SimVascular) — vascular flow basis adaptable to airways; PALABOS (https://gitlab.com/unigespc/palabos) — LBM for alveolar flow; 3D Slicer + SlicerMorph (https://github.com/SlicerMorph/SlicerMorph) — airway segmentation from CT.
- **CUDA libraries & GPU pattern:** CUDA Thrust for particle sort/bin operations; custom CUDA kernels for Lagrangian force integration (one thread per particle); cuSPARSE for airflow linear solve; pattern: dual-stream approach—Eulerian fluid on one SM partition, Lagrangian particles on another with atomic-add deposition counters.

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
