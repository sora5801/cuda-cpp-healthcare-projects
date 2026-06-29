# THEORY — 6.22 Bone Remodeling Simulation

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

### 6.22 Bone Remodeling Simulation 🟡 · Active R&D
- **Deep dive:** Bone continually remodels in response to mechanical loading: osteoclasts resorb bone and osteoblasts form new bone in a coupled feedback loop mediated by RANKL/OPG signaling. GPU simulation enables voxel-level finite element analysis of trabecular bone microstructure (µCT at 10–50 µm resolution yields 10⁸ voxels) and tracking remodeling over years of simulated time. Topology optimization algorithms (SIMP) on GPU-FEM underlie both bone remodeling models and prosthesis design.
- **Key algorithms:** Mechano-regulation theory (Prendergast/Huiskes), local strain energy density (SED) remodeling rule, cellular automata bone remodeling, RANKL/OPG ODE signaling network, nonlinear FEM for bone microstructure, SIMP topology optimization, homogenization for apparent stiffness.
- **Datasets:** PhysioNet bone-related datasets (https://physionet.org); OsteoArthritis Initiative (OAI) µCT and radiograph dataset (https://nda.nih.gov/oai/); BoneJ plugin morphometric datasets (https://bonej.org); MICCAI bone segmentation challenge datasets (grand-challenge.org).
- **Starter repos/tools:** FEBio (https://github.com/febiosoftware/FEBio) — nonlinear FEM for bone and cartilage; FreeFEM++ GPU extensions (verify URL) — PDE solver adaptable to remodeling; VoxFEM (verify URL — GPU voxel FEM from ETH Zurich research group); OpenFOAM for fluid-structure poroelastic bone (https://github.com/OpenFOAM/OpenFOAM-dev).
- **CUDA libraries & GPU pattern:** Custom CUDA kernel for per-voxel SED computation; cuSPARSE for voxel FEM assembly (structured sparse); cuSOLVER PCG for linear system; pattern: 3D CUDA thread grid matching voxel layout, shared memory for element stiffness assembly.

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
