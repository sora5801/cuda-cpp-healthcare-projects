# 10.17 — AR/VR Surgical Visualization & Real-Time Volume Rendering

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Biomechanics%2C%20Biomedical%20Devices%20%26%20Surgery-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 10: Biomechanics, Biomedical Devices & Surgery · Catalog ID `10.17`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

<!-- =======================================================================
     SCAFFOLD STATUS: this README was stamped from the catalog. The prose
     fields below (Deep dive / Algorithms / Datasets / Prior art) are filled
     in from the catalog. Sections marked TODO(impl)/TODO(theory) must be
     completed by the project author before this project is "done"
     (see CLAUDE.md §4.1 and tools/verify_project.py).
     ======================================================================= -->

## Summary

TODO(impl): One paragraph, plain language — what this project does and why a
learner should care. (Seed from the deep dive below.)

## What this computes & why the GPU helps

Augmented-reality surgical guidance requires sub-20 ms end-to-end latency from imaging sensor to rendered overlay, encompassing depth estimation, organ segmentation, tissue deformation tracking, and volume rendering on a single GPU. Ray-cast volume rendering of intraoperative ultrasound or cone-beam CT benefits from GPU empty-space skipping (sparse voxel octrees) and gradient-based shading. Neural rendering (NeRF / Gaussian splatting) trained on intraoperative images can reconstruct deforming organ surfaces in real time on an RTX GPU. The GPU parallelizes pixel-independent ray traversal, making volume rendering a textbook GPU workload with one thread per pixel.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Ray-cast volume rendering, gradient-magnitude transfer functions, sparse voxel octree traversal, NeRF / 3D Gaussian splatting for scene reconstruction, SLAM-based tracking, depth-from-stereo (disparity networks), mesh rasterization for AR overlay.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/ar-vr-surgical-visualization-real-time-volume-rendering.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/ar-vr-surgical-visualization-real-time-volume-rendering.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\ar-vr-surgical-visualization-real-time-volume-rendering.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, prints the result, shows the
GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/` — a tiny, offline input so the demo runs
  with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` (documented, idempotent).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: SciVis Contest Medical Volumes — benchmark CT/MR volumes for rendering (https://scivis.github.io/); SCARED stereo laparoscopy depth dataset (https://endovissub2019-scared.grand-challenge.org/); Hamlyn Robotic Vision Dataset (http://hamlyn.doc.ic.ac.uk/vision/); MICCAI 2023 Endoscopic Vision Challenge (verify URL via Grand Challenge).

## Expected output

Success looks like `demo/expected_output.txt`. The program computes the result on
both the **GPU** (`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`)
and asserts they agree within the documented tolerance — that agreement is the
correctness guarantee.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports.
2. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
3. [`src/kernels.cu`](src/kernels.cu) — the kernel(s) and host wrapper.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline.
5. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

NVIDIA CUDA-GL rendering samples (https://github.com/NVIDIA/cuda-samples) — volumerender sample; 3D Gaussian Splatting (https://github.com/graphdeco-inria/gaussian-splatting) — real-time neural rendering; VTK/vtkVolume (https://github.com/Kitware/VTK) — volume rendering with GPU acceleration; MONAI Label (https://github.com/Project-MONAI/MONAILabel) — real-time intraoperative segmentation.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CUDA texture objects (hardware-interpolated volume sampling), cuDNN for segmentation inference, OpenGL-CUDA interop for zero-copy display; pattern: intraoperative CT/US volume uploaded as 3D CUDA texture → one thread per display pixel ray-marches texture → alpha-compositing accumulation → OpenGL framebuffer blit → AR overlay. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
