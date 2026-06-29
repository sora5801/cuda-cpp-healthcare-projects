# THEORY — 2.9 Solvent-Accessible Surface & Poisson-Boltzmann Electrostatics

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

### 2.9 Solvent-Accessible Surface & Poisson-Boltzmann Electrostatics 🟢 · Established

- **Deep dive:** Continuum electrostatics models (Poisson-Boltzmann equation, PBE) compute the electrostatic potential of a protein in ionic solvent by solving a partial differential equation on a 3D grid. This enables calculation of protein pKa values, electrostatic binding contributions, and zeta potentials for colloidal drug carriers. GPU-accelerated PBE solvers (APBS, DelPhi-GPU) discretize the molecule onto a Eulerian grid and solve via Gauss-Seidel iteration or multigrid methods on GPU. The bottleneck is the 3D finite-difference PBE solve — parallelized via coloring (red-black ordering) on GPU threads.
- **Key algorithms:** Linearized Poisson-Boltzmann equation (LPBE), non-linear PBE, finite difference discretization (3D grid), red-black Gauss-Seidel iteration, multigrid preconditioning, generalized Born (GB) analytic approximation, SASA computation.
- **Datasets:** pKDBD — database of protein pKa values (verify URL); BindingMOAD — protein-ligand electrostatic data (https://bindingmoad.org); RCSB PDB structural data (https://www.rcsb.org); APBS validation benchmark (https://github.com/Electrostatics/apbs).
- **Starter repos/tools:** APBS (https://github.com/Electrostatics/apbs) — Poisson-Boltzmann solver with GPU acceleration; DelPhi (http://compbio.clemson.edu/delphi) — PB electrostatics with GPU solver; OpenMM GB force (https://github.com/openmm/openmm) — GPU Generalized Born; PDB2PQR (https://github.com/Electrostatics/pdb2pqr) — structure preparation for PBE.
- **CUDA libraries & GPU pattern:** CUDA thread blocks for 3D finite-difference red-black iteration; shared memory for stencil computation; cuSPARSE for sparse Laplacian matrix; GPU texture memory for dielectric boundary representation.

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
