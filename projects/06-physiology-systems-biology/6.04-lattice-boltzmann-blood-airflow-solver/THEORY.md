# THEORY — 6.4 Lattice-Boltzmann Blood/Airflow Solver

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

### 6.4 Lattice-Boltzmann Blood/Airflow Solver 🟡 · Active R&D
- **Deep dive:** The lattice-Boltzmann method (LBM) replaces continuum Navier-Stokes with a mesoscale kinetic equation for particle distribution functions on a regular grid—ideal for GPUs because each lattice site updates independently using only nearest-neighbor communication (the BGK collision step). Blood in complex vascular trees, red blood cell suspension rheology, and pulmonary airflow through bronchial trees all benefit from this approach. HemeLB achieves ~29.5 billion lattice site updates per second on thousands of cores; GPU versions (e.g., HemeLB GPU branch, PALABOS GPU) push throughput further with shared-memory streaming.
- **Key algorithms:** BGK (Bhatnagar-Gross-Krook) collision operator, multi-relaxation time (MRT) LBM, D3Q19/D3Q27 velocity stencils, bounce-back boundary conditions for no-slip walls, Shan-Chen multiphase LBM, immersed boundary method for red blood cell membranes, Palabos fluid-particle coupling.
- **Datasets:** PhysioNet coronary/aortic waveforms (https://physionet.org); Vascular Model Repository geometries (http://www.vascularmodel.com); open-access bronchial tree CT data from LIDC-IDRI (https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI); UK Biobank aortic flow MRI (https://www.ukbiobank.ac.uk).
- **Starter repos/tools:** HemeLB (https://github.com/hemelb-codes/hemelb) — sparse-geometry vascular LBM, MPI+GPU, scales to 32 000+ cores; HemePure GPU variant (https://github.com/hemelb-codes/HemePure) — cleaned GPU-first branch; PALABOS (https://gitlab.com/unigespc/palabos) — full-featured C++ LBM framework including multiphase and thermal extensions; USERMESO-2.0 (https://github.com/AnselGitAccount/USERMESO-2.0) — GPU red blood cell hemodynamics with deformable membrane.
- **CUDA libraries & GPU pattern:** Custom CUDA kernels for BGK streaming+collision in a single fused pass; shared memory for D3Q19 population arrays; texture memory for geometry masks; NCCL for GPU-direct halo exchange; pattern: one-thread-per-lattice-site with coalesced memory access on SOA layout.

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
