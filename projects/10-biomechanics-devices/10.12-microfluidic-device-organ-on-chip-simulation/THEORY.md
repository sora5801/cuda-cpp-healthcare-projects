# THEORY — 10.12 Microfluidic Device & Organ-on-Chip Simulation

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

### 10.12 Microfluidic Device & Organ-on-Chip Simulation 🟡 · Active R&D

- **Deep dive:** Lab-on-a-chip and organ-on-chip devices feature micrometer-scale channels where Re < 1 and Péclet numbers span orders of magnitude, demanding accurate Navier-Stokes + advection-diffusion solutions on geometrically complex domains. Lattice-Boltzmann Method (LBM) maps perfectly to GPU: each lattice node streams and collides independently, achieving memory-bandwidth-bound performance near GPU peak. GPU LBM-DEM (discrete element method) co-simulates cell transport, adhesion, and deformation through microchannels. Design optimization of pillar geometry, channel bifurcations, and gradient generators runs via adjoint sensitivity on GPU, drastically accelerating the design-of-experiment cycle for organ-chip platforms.
- **Key algorithms:** D3Q19/D3Q27 LBM with BGK or MRT collision, immersed boundary coupling for deformable cells, lattice-DEM for rigid particle transport, advection-diffusion for chemical gradient generation, adjoint sensitivity analysis for geometry optimization.
- **Datasets:** Microfluidic Gradient Generator Benchmark (LBM validation, Zenodo); PhysioMimetics organ-chip flow data (verify URL); OpenFOAM microfluidic validation cases (https://www.openfoam.com/); Glioblastoma-on-chip CFD dataset (Frontiers Bioeng 2025) (verify Zenodo).
- **Starter repos/tools:** Palabos (https://gitlab.com/unigespc/palabos) — GPU-capable LBM library for complex fluid dynamics; LEDDS (https://arxiv.org/abs/2512.04997) — portable LBM-DEM GPU simulations; waLBerla (https://www.walberla.net/) — massively parallel LBM framework with GPU support; OpenFOAM (https://github.com/OpenFOAM) — with GPU-accelerated linear solvers via PETSc-CUDA backend.
- **CUDA libraries & GPU pattern:** CUDA kernels for per-node stream-and-collide (one thread per lattice node), cuFFT for spectral pressure solve, Thrust for particle tracking; pattern: GPU-resident 3D lattice → CUDA stream-and-collide kernel → IBM force spreading for deformable cells → chemical concentration advection-diffusion update → device geometry optimization via adjoint.

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
