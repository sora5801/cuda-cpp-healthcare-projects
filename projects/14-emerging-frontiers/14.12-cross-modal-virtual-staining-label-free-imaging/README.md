# 14.12 — Cross-Modal "Virtual Staining" & Label-Free Imaging

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Emerging%2C%20Theoretical%20%26%20Grand--Challenge%20Frontiers-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 14: Emerging, Theoretical & Grand-Challenge Frontiers · Catalog ID `14.12`
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

Virtual staining uses deep learning to predict H&E, IHC, or other chemical stain images from label-free optical modalities (autofluorescence, quantitative phase, CARS, FTIR), eliminating destructive sample preparation. Pixel super-resolved virtual staining via diffusion models (Nature Communications 2025) achieves pathologist-grade tissue diagnostics from autofluorescence alone on GPU. GPU acceleration is essential: a single whole-slide image (100,000 × 100,000 pixels) requires tiled U-Net inference over ~10,000 patches per slide. Clinical-grade validation of autofluorescence virtual staining for prostate cancer (medRxiv 2024) demonstrates diagnostic equivalence to H&E. The GPU also enables real-time virtual staining during surgery for fresh frozen section replacement.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

U-Net/ViT image translation (pix2pix, CycleGAN, diffusion model), pixel super-resolution (ESRGAN, diffusion), Fourier ptychographic reconstruction, stimulated Raman spectral unmixing on GPU, multi-modal image registration (DRIT++), diffusion-model inversion for unpaired translation.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/cross-modal-virtual-staining-label-free-imaging.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/cross-modal-virtual-staining-label-free-imaging.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\cross-modal-virtual-staining-label-free-imaging.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Virtual Staining Dataset (Ozcan Lab, UCLA) — autofluorescence → H&E paired images (verify URL via nature.com supplementary); LCI-PARIS — unstained label-free vs. H&E pairs (verify URL); TCGA Digital Pathology Whole-Slide Images (https://portal.gdc.cancer.gov/); Human Protein Atlas — multimodal tissue images (https://www.proteinatlas.org/).

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

MONAI (https://github.com/Project-MONAI/MONAI) — GPU medical image segmentation + translation; pix2pix/CycleGAN (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) — paired/unpaired GPU image translation; Stable Diffusion (huggingface) fine-tuned for pathology virtual staining; HistoStar (https://github.com/TissueImageAnalytics/tiatoolbox) — GPU whole-slide image analysis toolkit.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuDNN for U-Net/ViT inference, Tensor Core FP16 for batch patch processing, cuFFT for Fourier ptychographic phase reconstruction; pattern: WSI tiled into 256×256 patches → GPU batch U-Net inference → tile stitching → GPU super-resolution → virtual H&E output for pathologist review. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
