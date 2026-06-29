# THEORY — 10.17 AR/VR Surgical Visualization & Real-Time Volume Rendering

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

### 10.17 AR/VR Surgical Visualization & Real-Time Volume Rendering 🟡 · Active R&D

- **Deep dive:** Augmented-reality surgical guidance requires sub-20 ms end-to-end latency from imaging sensor to rendered overlay, encompassing depth estimation, organ segmentation, tissue deformation tracking, and volume rendering on a single GPU. Ray-cast volume rendering of intraoperative ultrasound or cone-beam CT benefits from GPU empty-space skipping (sparse voxel octrees) and gradient-based shading. Neural rendering (NeRF / Gaussian splatting) trained on intraoperative images can reconstruct deforming organ surfaces in real time on an RTX GPU. The GPU parallelizes pixel-independent ray traversal, making volume rendering a textbook GPU workload with one thread per pixel.
- **Key algorithms:** Ray-cast volume rendering, gradient-magnitude transfer functions, sparse voxel octree traversal, NeRF / 3D Gaussian splatting for scene reconstruction, SLAM-based tracking, depth-from-stereo (disparity networks), mesh rasterization for AR overlay.
- **Datasets:** SciVis Contest Medical Volumes — benchmark CT/MR volumes for rendering (https://scivis.github.io/); SCARED stereo laparoscopy depth dataset (https://endovissub2019-scared.grand-challenge.org/); Hamlyn Robotic Vision Dataset (http://hamlyn.doc.ic.ac.uk/vision/); MICCAI 2023 Endoscopic Vision Challenge (verify URL via Grand Challenge).
- **Starter repos/tools:** NVIDIA CUDA-GL rendering samples (https://github.com/NVIDIA/cuda-samples) — volumerender sample; 3D Gaussian Splatting (https://github.com/graphdeco-inria/gaussian-splatting) — real-time neural rendering; VTK/vtkVolume (https://github.com/Kitware/VTK) — volume rendering with GPU acceleration; MONAI Label (https://github.com/Project-MONAI/MONAILabel) — real-time intraoperative segmentation.
- **CUDA libraries & GPU pattern:** CUDA texture objects (hardware-interpolated volume sampling), cuDNN for segmentation inference, OpenGL-CUDA interop for zero-copy display; pattern: intraoperative CT/US volume uploaded as 3D CUDA texture → one thread per display pixel ray-marches texture → alpha-compositing accumulation → OpenGL framebuffer blit → AR overlay.

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
