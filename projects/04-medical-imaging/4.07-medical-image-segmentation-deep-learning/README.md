# 4.7 — Medical Image Segmentation (Deep Learning)

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟢 Beginner · Established** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.7`
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

Volumetric segmentation of anatomical structures (organs, tumors, vessels) in CT/MRI using encoder-decoder CNNs operates on 3D patches or whole volumes; a 512×512×200 CT volume processed in a 3D U-Net with standard batch size requires ~16 GB GPU memory and ~200 GFLOPS per forward pass. nnU-Net automatically configures patch size, batch size, network topology, and augmentation to dataset fingerprints, making it a strong universal baseline. Inference of whole-body CT (TotalSegmentator, 117 structures) completes in 20–50 s on a GPU vs. 40–50 min on CPU. Transformer architectures (Swin-UNETR) add self-attention with quadratic memory cost in sequence length, further motivating large-VRAM GPUs.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

3D U-Net, nnU-Net, Swin-UNETR, TransUNet, DeepMedic, V-Net, residual encoder-decoder, cascaded networks, multi-scale feature pyramid, conditional random fields (CRF) post-processing, semi-supervised pseudo-labeling.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/medical-image-segmentation-deep-learning.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/medical-image-segmentation-deep-learning.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\medical-image-segmentation-deep-learning.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Medical Segmentation Decathlon (http://medicaldecathlon.com/) — 10 tasks, ~2,500 volumes total; TotalSegmentator training set (Zenodo, ~1,200 CT with 117 structure labels; https://zenodo.org/record/6802614); KiTS23 kidney tumor challenge (https://kits-challenge.org/kits23/); BraTS brain tumor dataset (https://www.synapse.org/#!Synapse:syn27046444).

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

nnU-Net (https://github.com/MIC-DKFZ/nnUNet) — self-configuring, handles 2D/3D, GPU training and inference; TotalSegmentator (https://github.com/wasserth/TotalSegmentator) — 117-class whole-body CT segmentation, GPU inference in <1 min; MONAI (https://github.com/Project-MONAI/MONAI) — PyTorch medical AI framework with GPU-optimized transforms and network zoo; Swin-UNETR reference (https://github.com/Project-MONAI/research-contributions) — transformer-based 3D segmentation.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuDNN (3D convolutions), Tensor Cores (FP16/BF16), CUDA Unified Memory for large volumes; mixed-precision training; patch-based inference with sliding window; multi-GPU via PyTorch DDP + NCCL; GPU-resident data augmentation (random flips, elastic deformations) via MONAI or NVIDIA DALI. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
