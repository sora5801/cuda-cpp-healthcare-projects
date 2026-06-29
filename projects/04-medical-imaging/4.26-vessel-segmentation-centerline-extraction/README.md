# 4.26 — Vessel Segmentation & Centerline Extraction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.26`
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

Vascular tree segmentation in CT angiography (CTA) detects tubular structures as small as 1–2 mm diameter in noisy 3D volumes; GPU-accelerated Hessian-based vesselness filters (Frangi) compute the full 3×3 Hessian eigenvalue decomposition per voxel — ~10⁶ symmetric 3×3 Eigen-decompositions for a clinical CTA. U-Net-based vessel segmentation processes the full 3D volume in overlapping patches, requiring GPU for interactive-speed inference. Centerline extraction via fast-marching or geodesic path algorithms is inherently sequential but GPU implementations exist via parallel priority queues. Clinical applications include coronary CTA FFRCT computation and aortic endograft planning.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Hessian-based vesselness filter (Frangi, Sato), multi-scale vesselness (scale-space), 3D U-Net vessel segmentation, V-Net, nnDetection for tubular object detection, fast-marching centerline (FMM on GPU), minimum-path centerline (Dijkstra-like), vascular topology graph extraction.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/vessel-segmentation-centerline-extraction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/vessel-segmentation-centerline-extraction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\vessel-segmentation-centerline-extraction.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: ASOCA (Automated Segmentation of Coronary Arteries, https://asoca.grand-challenge.org/); VesselMAP (cerebral vessels, verify URL); IRCAD 3D-IRCADb-01 abdominal (https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/); ImageCAS coronary artery dataset (https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT).

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

VMTK (Vascular Modeling Toolkit, https://github.com/vmtk/vmtk) — centerline extraction, meshing, CFD integration; SlicerVMTK (https://github.com/vmtk/SlicerExtension-VMTK) — 3D Slicer integration; MONAI (https://github.com/Project-MONAI/MONAI) — 3D vessel segmentation networks; nnDetection (https://github.com/MIC-DKFZ/nnDetection) — GPU object detection for tubular structures.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom CUDA Hessian kernel (per-voxel 3×3 eigendecomposition using Jacobi iteration); cuDNN (3D U-Net inference); GPU priority queue for parallel fast-marching (thrust); shared memory for neighborhood gradient computation. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
