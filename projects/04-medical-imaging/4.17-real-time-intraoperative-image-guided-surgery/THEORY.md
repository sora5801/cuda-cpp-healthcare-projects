# THEORY — 4.17 Real-Time Intraoperative / Image-Guided Surgery

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

### 4.17 Real-Time Intraoperative / Image-Guided Surgery 🟡 · Active R&D
- **Deep dive:** Image-guided surgery (IGS) fuses preoperative MRI/CT with intraoperative imaging (ultrasound, CBCT, fluorescence) to track surgical instruments and tumor margins in real time. The latency budget is <100 ms for tool tracking and <1 s for image update. GPU acceleration is required at every stage: intraoperative CBCT reconstruction (FDK in <1 s), deformable registration of pre/intra-operative volumes (<5 s), instrument segmentation from camera or US feed (<50 ms/frame), and DRR generation for X-ray/CT registration (<20 ms). Brain shift correction requires deformable surface registration incorporating intraoperative US and biomechanical models, solvable via GPU finite-element methods.
- **Key algorithms:** GPU FDK (CBCT intraoperative), Iterated closest point (ICP) for surface registration, GPU Demons for deformable brain-shift correction, CNN-based instrument segmentation (U-Net, YOLOv8), neural radiance fields (NeRF) for surgical scene reconstruction, Kalman filtering for tool tracking.
- **Datasets:** Cholec80 laparoscopic video dataset (https://camma.u-strasbg.fr/datasets); ReMIND2Reg 2025 brain resection multimodal dataset (https://arxiv.org/abs/2508.09649); EndoVis MICCAI challenge datasets (https://endovis.grand-challenge.org/); SurgT benchmark for surgical tool tracking.
- **Starter repos/tools:** PLUS (Public Software Library for Ultrasound Imaging Research, https://github.com/PlusToolkit/PlusLib) — real-time US acquisition/reconstruction; 3D Slicer (https://github.com/Slicer/Slicer) — OpenIGTLink for intraoperative GPU-accelerated 3D rendering; NVIDIA Clara Holoscan (https://github.com/nvidia-holoscan/holoscan-sdk) — real-time medical imaging SDK with GPU pipeline; RTK (https://github.com/RTKConsortium/RTK) — intraoperative CBCT reconstruction.
- **CUDA libraries & GPU pattern:** cuFFT + custom CUDA FDK for sub-second CBCT; cuBLAS for ICP normal-equation solve; cuDNN for instrument seg CNN inference; CUDA OpenGL interop for real-time 3D visualization overlay; NVIDIA Holoscan pipeline for <10 ms latency.

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
