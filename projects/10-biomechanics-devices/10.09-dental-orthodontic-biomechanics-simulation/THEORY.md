# THEORY — 10.9 Dental & Orthodontic Biomechanics Simulation

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

### 10.9 Dental & Orthodontic Biomechanics Simulation 🟡 · Active R&D

- **Deep dive:** Orthodontic tooth movement depends on PDL (periodontal ligament) stress distribution, alveolar bone remodeling, and contact forces between brackets, wires, and clear aligners — all requiring nonlinear FEA on individually segmented CBCT geometries. GPU acceleration allows the dense contact constraint systems (dozens of tooth-aligner contact pairs per timestep) to be assembled and solved in parallel, enabling treatment planning that runs in minutes rather than hours. Dental implant osseointegration modeling couples elastic bone FEM with a poroelastic fluid-in-pore submodel at the implant interface. Population-scale virtual clinical trials — thousands of patient-specific models run simultaneously — become feasible on GPU clusters.
- **Key algorithms:** Hyperelastic PDL material models (Mooney-Rivlin, Ogden), bone-remodeling (Frost mechanostat), penalty-based contact, mortar contact formulation, thermo-mechanical coupling for composite restorations, coupled poroelastic FEM.
- **Datasets:** CBCT Tooth Segmentation Challenge (ToothFairy, MICCAI 2023) — annotated dental CBCT (https://toothfairy.grand-challenge.org/); 3D Dental Mesh Dataset (Teeth3DS) — 1800 intraoral scans (https://github.com/abenhamadou/3DTeethSeg22_challenge); NIH NIDCR FaceBase craniofacial CT atlas (https://www.facebase.org/); Open Dental Science datasets — clinical records + x-rays (verify URL via opendentalsoftware.com).
- **Starter repos/tools:** FEBio (https://github.com/febiosoftware/FEBio) — handles PDL and bone-remodeling constitutive models; CGAL (https://github.com/CGAL/cgal) — mesh generation from CBCT segmentations; ITK-SNAP (https://www.itksnap.org/) — CBCT segmentation to mesh pipeline; 3DTeethSeg (https://github.com/abenhamadou/3DTeethSeg22_challenge) — tooth segmentation model for mesh generation.
- **CUDA libraries & GPU pattern:** cuSPARSE/cuSolver for contact-augmented stiffness matrix, CUDA kernels for per-element PDL stress update, Thrust for mortar contact pair enumeration; pattern: element-parallel stiffness assembly → penalty contact augmentation → PCG solve on GPU → bone-density update → geometry export for aligner CAD.

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
