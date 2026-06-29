# 4.18 — Image-Based 3D Printing / Model Generation for Surgery

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟢 Beginner · Established** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.18`
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

Patient-specific anatomical models for surgical rehearsal require segmenting CT/MRI volumes (GPU CNN inference), smoothing and decimating meshes (GPU geometry processing), and generating printable STL/OBJ files. For a full torso CT at 0.5 mm isotropic resolution the input volume is ~10⁹ voxels; running marching cubes on GPU (NVIDIA CUB-accelerated or CUDA-native) reduces the surface extraction step from minutes to seconds. Multi-material prints (bone, soft tissue, vessels) require multi-label segmentation and per-label mesh Boolean operations — all benefiting from GPU parallelism. Finite-element simulation for patient-specific implant design (titanium plates, aortic stents) additionally uses GPU FEM solvers.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

GPU marching cubes (isosurface extraction), mesh smoothing (Laplacian, Taubin), Boolean mesh operations, multi-material voxel-to-mesh, TotalSegmentator CNN segmentation, GPU FEM (finite element method) for biomechanics, support structure generation for FDM printing.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/image-based-3d-printing-model-generation-for-surgery.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/image-based-3d-printing-model-generation-for-surgery.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\image-based-3d-printing-model-generation-for-surgery.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: TCIA body CT collections; OsteoArthritis Initiative (OAI) for knee models (https://nda.nih.gov/oai/); VerSe vertebral segmentation dataset (https://github.com/anjany/verse); TotalSegmentator dataset (https://zenodo.org/record/6802614).

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

3D Slicer (https://github.com/Slicer/Slicer) — GPU-accelerated volume rendering, segmentation, STL export via SlicerRT; VTK (https://vtk.org/) — GPU-accelerated marching cubes and mesh operations; TotalSegmentator (https://github.com/wasserth/TotalSegmentator) — fast GPU segmentation for print-ready model prep; OpenVDB (https://www.openvdb.org/) — GPU sparse volume processing for complex anatomies.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CUDA marching cubes (thrust scan for compact output); cuBLAS for FEM stiffness matrix assembly; GPU ray-casting for volume rendering overlay; custom CUDA for Laplacian smoothing (per-vertex neighbor average); cuSPARSE for FEM linear system. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
