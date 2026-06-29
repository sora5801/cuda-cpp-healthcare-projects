# THEORY — 10.8 Computational Fluid-Structure Interaction for Devices

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

### 10.8 Computational Fluid-Structure Interaction for Devices 🟡 · Active R&D

- **Deep dive:** Heart valves, stents, LVADs, and arterial stents involve tightly coupled incompressible fluid (blood) and elastic/rigid solid (leaflets, walls) dynamics that must be co-solved. Immersed boundary methods (IBM) embed flexible structures in Eulerian fluid grids, requiring interpolation and spreading operations that are GPU-parallelized across boundary points. SPH (smoothed particle hydrodynamics) replaces grids with Lagrangian particles, enabling free-surface and high-deformation flows suitable for LVAD impeller modeling. The FSEI-GPU code solves fluid-structure-electrophysiology interaction of the full left heart on a few GPU cards, completing one heartbeat in hours instead of days. Multi-GPU domain decomposition via NCCL enables scaling to whole-cardiovascular-system models.
- **Key algorithms:** Immersed Boundary Method (IBM), Lattice-Boltzmann Method (LBM), ISPH/TLSPH Smoothed Particle Hydrodynamics, arbitrary Lagrangian-Eulerian (ALE) formulation, Navier-Stokes fractional-step solver, hemolysis (GKM model) and thrombosis (biochemical agonist) submodels.
- **Datasets:** 4D Flow MRI Benchmark (HEArt) — time-resolved 3D velocity fields in cardiac chambers (https://arxiv.org/abs/2111.00720); HeartFlow FFRCT coronary dataset (commercial, academic access); Aortic Flow Simulation Database from SimVascular (https://simvascular.github.io/); OpenHeart MRI cohort — segmented cardiac geometries (verify URL via Zenodo).
- **Starter repos/tools:** FSEI-GPU (https://arxiv.org/abs/2103.15187) — CUDA Fortran FSI+electrophysiology heart solver (see ScienceDirect for code link); SimVascular (https://github.com/SimVascular/SimVascular) — patient-specific cardiovascular FSI pipeline; GPU-accelerated IB solver (Bhalla group, https://arxiv.org/html/2605.04335) — OpenACC + CUDA + NCCL extreme-scale IBM; PyFR (https://github.com/PyFR/PyFR) — GPU-native high-order Navier-Stokes solver adaptable to biofluid domains.
- **CUDA libraries & GPU pattern:** CUDA kernels for IBM force-spreading/interpolation, cuFFT for Poisson pressure solve, NCCL for multi-GPU halo exchange, cuSPARSE for FSI coupling matrix; pattern: Eulerian fluid grid partitioned across GPUs → IBM Lagrangian marker forces spread to fluid grid via CUDA kernel → pressure solve via FFT → structure positions updated → halo exchange via NCCL.

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
