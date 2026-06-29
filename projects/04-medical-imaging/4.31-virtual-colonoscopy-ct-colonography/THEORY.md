# THEORY — 4.31 Virtual Colonoscopy & CT Colonography

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

### 4.31 Virtual Colonoscopy & CT Colonography 🟢 · Established
- **Deep dive:** CT colonography (CTC) acquires a supine/prone CT of the air-distended colon, then generates a virtual endoscopic fly-through rendered from inside the colonic lumen. The rendering pipeline involves: (1) colon segmentation from 512³ CT (GPU CNN inference), (2) electronic colon cleansing, (3) colonic centerline extraction (GPU fast-marching), (4) real-time volume rendering of the lumen interior (GPU ray-casting), and (5) computer-aided polyp detection (CNN classifier on rendered views or 3D patches). Real-time fly-through at 60 FPS requires GPU-accelerated volume rendering; polyp detection on an annotated virtual endoscopy dataset requires GPU training and inference.
- **Key algorithms:** GPU volume ray-casting (gradient-magnitude + Phong shading), electronic colon cleansing (thin-plate spline tagged material subtraction), centerline fast-marching, nnU-Net colon segmentation, 3D CNN polyp detection, CTC U-Net for lumen segmentation, shape index / curvedness for polyp candidates.
- **Datasets:** TCIA CT Colonography dataset (https://wiki.cancerimagingarchive.net/display/Public/CT+Colonography); MICCAI 2018 colon challenge; ACR Radiology Lung-RADS CT dataset; NLST CT colonography subsets.
- **Starter repos/tools:** 3D Slicer (https://github.com/Slicer/Slicer) — GPU volume rendering and colon seg module; VTK (https://vtk.org/) — GPU volume ray-casting engine; MONAI (https://github.com/Project-MONAI/MONAI) — nnU-Net colon segmentation; VisIt (https://visit-dav.github.io/visit-website/) — GPU visualization for large CT volumes.
- **CUDA libraries & GPU pattern:** CUDA OptiX / OpenGL ray-casting with volume texture; custom CUDA for gradient magnitude (Sobel, per-voxel 26-neighbor); cuDNN for polyp detection CNN; CUDA 3D texture for fast trilinear lookup during fly-through; GPU-resident colon mesh for real-time rendering.

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
