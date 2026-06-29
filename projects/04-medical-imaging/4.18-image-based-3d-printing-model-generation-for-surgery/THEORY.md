# THEORY — 4.18 Image-Based 3D Printing / Model Generation for Surgery

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

### 4.18 Image-Based 3D Printing / Model Generation for Surgery 🟢 · Established
- **Deep dive:** Patient-specific anatomical models for surgical rehearsal require segmenting CT/MRI volumes (GPU CNN inference), smoothing and decimating meshes (GPU geometry processing), and generating printable STL/OBJ files. For a full torso CT at 0.5 mm isotropic resolution the input volume is ~10⁹ voxels; running marching cubes on GPU (NVIDIA CUB-accelerated or CUDA-native) reduces the surface extraction step from minutes to seconds. Multi-material prints (bone, soft tissue, vessels) require multi-label segmentation and per-label mesh Boolean operations — all benefiting from GPU parallelism. Finite-element simulation for patient-specific implant design (titanium plates, aortic stents) additionally uses GPU FEM solvers.
- **Key algorithms:** GPU marching cubes (isosurface extraction), mesh smoothing (Laplacian, Taubin), Boolean mesh operations, multi-material voxel-to-mesh, TotalSegmentator CNN segmentation, GPU FEM (finite element method) for biomechanics, support structure generation for FDM printing.
- **Datasets:** TCIA body CT collections; OsteoArthritis Initiative (OAI) for knee models (https://nda.nih.gov/oai/); VerSe vertebral segmentation dataset (https://github.com/anjany/verse); TotalSegmentator dataset (https://zenodo.org/record/6802614).
- **Starter repos/tools:** 3D Slicer (https://github.com/Slicer/Slicer) — GPU-accelerated volume rendering, segmentation, STL export via SlicerRT; VTK (https://vtk.org/) — GPU-accelerated marching cubes and mesh operations; TotalSegmentator (https://github.com/wasserth/TotalSegmentator) — fast GPU segmentation for print-ready model prep; OpenVDB (https://www.openvdb.org/) — GPU sparse volume processing for complex anatomies.
- **CUDA libraries & GPU pattern:** CUDA marching cubes (thrust scan for compact output); cuBLAS for FEM stiffness matrix assembly; GPU ray-casting for volume rendering overlay; custom CUDA for Laplacian smoothing (per-vertex neighbor average); cuSPARSE for FEM linear system.

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
