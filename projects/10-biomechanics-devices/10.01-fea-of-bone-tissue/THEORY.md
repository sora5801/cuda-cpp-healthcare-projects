# THEORY — 10.1 FEA of Bone & Tissue

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

### 10.1 FEA of Bone & Tissue 🟢 · Established

- **Deep dive:** Finite-element analysis of bone and soft tissue solves systems of millions of coupled equations relating stress, strain, and material nonlinearity under physiological loading. GPU parallelism targets the sparse-matrix assembly and iterative linear-solver (conjugate-gradient or multigrid) phases, which dominate wall time in large 3D meshes. Co-rotational and total-Lagrangian explicit dynamics (TLED) formulations map naturally to SIMT execution because each element's stiffness update is independent. Bone-remodeling simulations (Wolff's law) couple mechanical fields with density update rules, requiring repeated solve-update-resolve cycles that each benefit from CUDA acceleration. Real-world targets include vertebral fracture prediction, hip-implant stress-shielding, and micro-CT-derived trabecular models with >10 M elements.
- **Key algorithms:** Total Lagrangian Explicit Dynamics (TLED), co-rotational FEM, neo-Hookean / Mooney-Rivlin hyperelasticity, preconditioned conjugate gradient (PCG) with Jacobi or incomplete-Cholesky preconditioners, bone-remodeling (Beaupré–Carter) adaptation loops.
- **Datasets:** FEBio Benchmark Suite — verified test problems for nonlinear biomechanical FEA (https://febio.org/knowledgebase/); Open Knee(s) — subject-specific knee joint FE models with segmented cartilage/bone (https://simtk.org/projects/openknee); Visible Human Project — full CT/MRI cadaver data for mesh generation (https://www.nlm.nih.gov/research/visible/visible_human.html); Bone-Load Database (Bergmann et al.) — in vivo implant load telemetry for hip and knee (https://orthoload.com/).
- **Starter repos/tools:** FEBio (https://github.com/febiosoftware/FEBio) — open-source nonlinear FE solver for biomechanics, C++, with GPU-solver hooks; NiftySim (https://github.com/eloygarcia/niftysim) — CUDA TLED soft-tissue FE toolkit from UCL; NVIDIA CUDALibrarySamples (https://github.com/NVIDIA/CUDALibrarySamples) — cuSPARSE/cuSolver conjugate-gradient templates; Awesome-Biomechanics (https://github.com/modenaxe/awesome-biomechanics) — curated dataset/tool index.
- **CUDA libraries & GPU pattern:** cuSPARSE (SpMV in PCG inner loop), cuSolver (direct sparse factorization), cuBLAS (dense BLAS), Thrust (parallel reductions); pattern: one CUDA thread per element for stiffness assembly → global atomic scatter into CSR matrix → iterative solver in cuSPARSE.

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
