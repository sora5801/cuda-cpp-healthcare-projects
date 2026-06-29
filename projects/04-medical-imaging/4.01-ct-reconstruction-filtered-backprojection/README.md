# 4.1 — CT Reconstruction — Filtered Backprojection

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟢 Beginner · Established** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.1`
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

Computes a 3D volume from a set of 2D X-ray projections by applying a ramp (Ram-Lak) filter in the frequency domain to each sinogram row, then smearing each filtered projection back across the reconstructed volume. The Feldkamp-Davis-Kress (FDK) algorithm extends this to cone-beam geometry used in modern scanners and linac on-board imagers. GPU acceleration is decisive: for a 512³ volume and 1,000 projections, each backprojection step touches ~10⁹ voxel-projection pairs, making serial CPU execution intractable for real-time or high-resolution use. CUDA texture memory provides hardware-interpolated trilinear sampling of projection data at near-zero extra cost, and the entire backprojection kernel saturates GPU memory bandwidth. Achieving sub-second reconstruction at clinical resolutions requires tens of TFLOPS, available only on GPU.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Feldkamp-Davis-Kress FBP, Ram-Lak / Shepp-Logan ramp filter, Parker short-scan weighting, GPU ray-driven and voxel-driven backprojection, helical cone-beam FDK with Katsevich exact reconstruction.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/ct-reconstruction-filtered-backprojection.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/ct-reconstruction-filtered-backprojection.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\ct-reconstruction-filtered-backprojection.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: LUNA16/LIDC-IDRI — 888 annotated thoracic CTs from TCIA (https://luna16.grand-challenge.org/); TCIA (The Cancer Imaging Archive) — large multi-collection public CT/MRI archive (https://www.cancerimagingarchive.net/); LoDoPaB-CT — low-dose CT sinogram/reconstruction pairs for benchmarking (https://zenodo.org/record/3384092); 2016 AAPM Low-Dose CT Grand Challenge — paired full-/quarter-dose CT scans (https://www.aapm.org/grandchallenge/lowdosect/).

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

RTK (RTKConsortium/RTK, https://github.com/RTKConsortium/RTK) — ITK-based, GPU FDK and iterative, multi-GPU, clinical DICOM-RT support; ASTRA Toolbox (https://astra-toolbox.com/, https://github.com/astra-toolbox/astra-toolbox) — MATLAB/Python/C++ GPU forward/back-projection primitives for 2D/3D, supports fan/cone/parallel; TIGRE (https://github.com/CERN/TIGRE) — MATLAB/Python CUDA toolbox with FDK plus 10+ iterative algorithms, real-dataset focus; Plastimatch (https://plastimatch.org/) — GPU FDK, deformable registration, DRR; open-source, clinical-grade C++.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuFFT (ramp filter in k-space), CUDA texture memory (hardware trilinear backprojection interpolation), cuBLAS; kernel pattern: one CUDA thread per output voxel, loops over projections; multi-GPU split over projection subsets. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
