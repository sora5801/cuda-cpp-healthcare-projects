# THEORY — 10.11 Cell-Membrane & Microstructural Mechanics

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

### 10.11 Cell-Membrane & Microstructural Mechanics 🟡 · Active R&D

- **Deep dive:** Red blood cell (RBC) deformability, cancer-cell invasion, and vesicle dynamics are governed by membrane bending elasticity (Helfrich model), cytoskeletal spectrin-network stretching, and viscous fluid-membrane coupling. GPU-accelerated dissipative particle dynamics (DPD) or multi-GPU LBM-IBM simulates thousands of RBCs simultaneously in microchannel flows, capturing population-level distributions of deformability index that are diagnostically relevant. Molecular dynamics of membrane lipid bilayers (GROMACS on GPU) resolves pore formation during electroporation or drug insertion. The bottleneck is the N-body neighbor-list update at each timestep, parallelized via CUDA cell-lists.
- **Key algorithms:** Dissipative particle dynamics (DPD), spectrin-link spring network for RBC cytoskeleton, Helfrich bending elasticity, LBM-IBM coupling, coarse-grained MD (MARTINI force field), Monte Carlo moves for lipid flip-flop.
- **Datasets:** RBC deformability measurements (ektacytometry), DIADEM microfluidic datasets (verify URL); RCSB PDB lipid bilayer structures for MD initialization; Red Cell Project DPD parameter database (verify URL via pubs.acs.org); OpenRBC benchmark — large-scale DPD RBC simulations (verify URL).
- **Starter repos/tools:** GROMACS (https://github.com/gromacs/gromacs) — GPU MD with CUDA/HIP backend, supports CG membrane models; OpenRBC (https://github.com/pnnl/OpenRBC) — massively parallel DPD red blood cell simulator; LAMMPS (https://github.com/lammps/lammps) — GPU-accelerated MD/DPD with many membrane force fields; HemeLB (https://github.com/UCL/hemelb) — LBM blood flow with deformable cell coupling.
- **CUDA libraries & GPU pattern:** CUDA cell-list neighbor search, cuBLAS for force accumulation, NCCL for multi-GPU domain decomposition; pattern: spatial cell-list on GPU → O(N) pair-force evaluation in CUDA kernels → Verlet or velocity-Störmer integrator → periodic boundary halo exchange via NCCL.

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
