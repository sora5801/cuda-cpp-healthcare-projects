# THEORY — 10.3 Implant & Prosthetic Design Optimization

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

### 10.3 Implant & Prosthetic Design Optimization 🟡 · Active R&D

- **Deep dive:** Patient-specific implants (hip, knee, spinal, dental) require iterative structural optimization over high-resolution 3D voxel grids (>1 M elements), where density or level-set fields evolve based on sensitivity analysis from repeated FEA solves. GPU acceleration makes three-dimensional SIMP (Solid Isotropic Material with Penalization) topology optimization tractable: a single density update pass over a 256³ grid requires ~16 M stiffness evaluations that execute in parallel. Lattice-structure implants for osseointegration require multiscale homogenization, computing effective elastic tensors for thousands of unit-cell configurations in parallel on GPU. Bone-remodeling feedback loops then validate implant geometry by simulating load transfer over years of use.
- **Key algorithms:** SIMP topology optimization, density-based level-set method, homogenization of periodic lattices, finite-element sensitivity analysis, optimality criteria (OC) update, bone-remodeling (Weinans/Beaupré) adaptation, multi-objective Pareto optimization.
- **Datasets:** OrthoLoad Implant Loading Database — in vivo hip/knee/spine implant force telemetry (https://orthoload.com/); MICCAI 2023 VerSe Challenge — vertebral shape dataset for spinal implant design (https://verse-challenge.github.io/); Hip Implant Topology Dataset — validated micro-FE lattice endoprostheses (see https://www.nature.com/articles/s41598-024-56327-4); FDA Orthopaedic Simulator Database — standardized fatigue loading profiles (verify URL via FDA.gov).
- **Starter repos/tools:** GPU-Accelerated Topology Optimization (Paulino group, Princeton) (https://paulino.princeton.edu/journal_papers/2013/SMO_13_TowardGPUAccelerated.pdf) — multigrid GPU SIMP reference implementation; Simple and Efficient GPU TO (https://www.sciencedirect.com/science/article/pii/S0045782523001676) — open-source GPU TO code from 2023 CMAME paper (verify repo link in supplementary); FEBio (https://github.com/febiosoftware/FEBio) — sensitivity analysis infrastructure; ToPy (https://github.com/williamhunter/topy) — Python 2D/3D TO (CPU, GPU-extensible reference).
- **CUDA libraries & GPU pattern:** cuSPARSE for repeated sparse FE solves, cuDNN for CNN-based TO surrogate acceleration, Thrust for parallel density-filter convolutions; pattern: element-parallel stiffness + sensitivity computation → parallel density update → GPU multigrid V-cycle for equilibrium solve.

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
