# 4.9 — Image Denoising & Restoration

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.9`
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

Medical images suffer from quantum noise (CT, PET, X-ray), thermal noise (MRI), and speckle (ultrasound). Deep denoising networks (DnCNN, RED-CNN for low-dose CT, Noise2Void for unsupervised fluorescence) process 2D or 3D patches through many conv layers, requiring substantial FLOPS and large GPU memory for 3D volumetric batches. Diffusion-model denoisers now achieve state-of-the-art perceptual quality but require iterative reverse-diffusion steps (50–1,000 denoising steps), each a full forward pass through a large UNet, making GPU mandatory. Non-learning methods (NLM, BM4D) have O(N²) complexity in voxel count, acceleratable via CUDA block-matching and nearest-neighbor search.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

DnCNN, RED-CNN (residual encoder-decoder CNN for low-dose CT), Noise2Void (N2V), Noise2Self, score-based diffusion denoising (DDPM, DDIM), BM3D/BM4D, non-local means (NLM), wavelet shrinkage, total variation denoising.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/image-denoising-restoration.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/image-denoising-restoration.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\image-denoising-restoration.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: 2016 AAPM Low-Dose CT Challenge (https://www.aapm.org/grandchallenge/lowdosect/) — quarter-dose / full-dose pairs; NLST (National Lung Screening Trial) via TCIA; Fluorescence Microscopy Noise Dataset (https://github.com/juglab/n2v) — for Noise2Void; SIDD smartphone noise dataset (image domain).

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

N2V / Noise2Void (https://github.com/juglab/n2v) — self-supervised GPU denoising for microscopy and MRI; MONAI model zoo — RED-CNN and DnCNN for CT; DnCNN PyTorch (https://github.com/cszn/DnCNN) — GPU-accelerated Gaussian denoiser; DiffusionMBIR (https://github.com/HJ-harry/DiffusionMBIR) — score-based diffusion for CT reconstruction/denoising.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuDNN (2D/3D dilated convolutions in DnCNN); custom CUDA for NLM block matching (each thread computes patch distance vs. all neighbors); cuBLAS for fully connected layers; FP16 inference via TensorRT for clinical deployment. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
