# 4.17 — Real-Time Intraoperative / Image-Guided Surgery

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.17`
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

Image-guided surgery (IGS) fuses preoperative MRI/CT with intraoperative imaging (ultrasound, CBCT, fluorescence) to track surgical instruments and tumor margins in real time. The latency budget is <100 ms for tool tracking and <1 s for image update. GPU acceleration is required at every stage: intraoperative CBCT reconstruction (FDK in <1 s), deformable registration of pre/intra-operative volumes (<5 s), instrument segmentation from camera or US feed (<50 ms/frame), and DRR generation for X-ray/CT registration (<20 ms). Brain shift correction requires deformable surface registration incorporating intraoperative US and biomechanical models, solvable via GPU finite-element methods.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

GPU FDK (CBCT intraoperative), Iterated closest point (ICP) for surface registration, GPU Demons for deformable brain-shift correction, CNN-based instrument segmentation (U-Net, YOLOv8), neural radiance fields (NeRF) for surgical scene reconstruction, Kalman filtering for tool tracking.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/real-time-intraoperative-image-guided-surgery.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/real-time-intraoperative-image-guided-surgery.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\real-time-intraoperative-image-guided-surgery.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Cholec80 laparoscopic video dataset (https://camma.u-strasbg.fr/datasets); ReMIND2Reg 2025 brain resection multimodal dataset (https://arxiv.org/abs/2508.09649); EndoVis MICCAI challenge datasets (https://endovis.grand-challenge.org/); SurgT benchmark for surgical tool tracking.

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

PLUS (Public Software Library for Ultrasound Imaging Research, https://github.com/PlusToolkit/PlusLib) — real-time US acquisition/reconstruction; 3D Slicer (https://github.com/Slicer/Slicer) — OpenIGTLink for intraoperative GPU-accelerated 3D rendering; NVIDIA Clara Holoscan (https://github.com/nvidia-holoscan/holoscan-sdk) — real-time medical imaging SDK with GPU pipeline; RTK (https://github.com/RTKConsortium/RTK) — intraoperative CBCT reconstruction.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuFFT + custom CUDA FDK for sub-second CBCT; cuBLAS for ICP normal-equation solve; cuDNN for instrument seg CNN inference; CUDA OpenGL interop for real-time 3D visualization overlay; NVIDIA Holoscan pipeline for <10 ms latency. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
