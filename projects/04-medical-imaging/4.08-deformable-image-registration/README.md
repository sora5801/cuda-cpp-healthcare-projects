# 4.8 — Deformable Image Registration

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.8`
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

Deformable image registration (DIR) estimates a dense displacement vector field (DVF) that maps a moving image to a fixed image, minimizing an image dissimilarity metric (NCC, NMI, SSD) subject to a regularization penalty (bending energy, diffusion). Classical optimization (Demons, B-spline free-form deformation) requires hundreds of gradient descent iterations on each voxel of a dense DVF — ~10⁹ parameters for a 256³ volume — making per-iteration GPU parallelism essential. Learning-based methods (VoxelMorph) infer the DVF in a single forward pass (<1 s GPU vs. 2+ hrs ANTs CPU), but training requires large GPU memory for 3D batch processing. LDDMM (Large Deformation Diffeomorphic Metric Mapping) adds geodesic shooting on the diffeomorphism group, computable via GPU-accelerated Fourier-domain operators.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Demons, diffeomorphic Demons, B-spline FFD (free-form deformation), LDDMM geodesic shooting, VoxelMorph (CNN-based), TransMorph (transformer-based), symmetric diffeomorphic normalization (SyN/ANTs), normalized cross-correlation (NCC) in GPU sliding-window.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/deformable-image-registration.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/deformable-image-registration.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\deformable-image-registration.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: OASIS brain MRI (https://www.oasis-brains.org/) — used in Learn2Reg challenge; Learn2Reg 2022 challenge (https://learn2reg.grand-challenge.org/) — lung, brain, abdominal; DIR-Lab lung CT deformation dataset (https://dir-lab.com/); 4D-CT lung datasets for respiratory motion.

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

VoxelMorph (https://github.com/voxelmorph/voxelmorph) — TF/PyTorch unsupervised GPU registration; Plastimatch (https://plastimatch.org/) — GPU B-spline and Demons, DICOM-RT support; ANTs (https://github.com/ANTsX/ANTs) — gold-standard SyN (CPU-only but widely used for ground truth); TransMorph (https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration) — Swin-transformer DIR, GPU-accelerated.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuFFT for LDDMM geodesic shooting; custom CUDA trilinear interpolation kernel for warp; cuBLAS for regularization Hessian; memory pattern: DVF and image volumes in GPU global memory; gradient computation via cuDNN autograd. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
