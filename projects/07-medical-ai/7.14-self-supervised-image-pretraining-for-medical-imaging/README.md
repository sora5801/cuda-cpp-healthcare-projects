# 7.14 — Self-Supervised Image Pretraining for Medical Imaging

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20AI%20%26%20Clinical%20Deep%20Learning-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 7: Medical AI & Clinical Deep Learning · Catalog ID `7.14`
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

Pre-trains visual encoders on large unlabelled medical image collections using contrastive or masked image modelling objectives, so downstream tasks (classification, segmentation) require only small labelled datasets. SimCLR, MoCo-v3, DINO, and MAE all reduce to large batched matrix multiplications during projection head training and attention; MoCo avoids the large-batch requirement of SimCLR by using a momentum encoder and GPU-resident queue of negatives. Medical images differ from natural images (greyscale, 3D, complex domain shifts), requiring domain-specific augmentation policies. GPU memory is the primary constraint: SimCLR needs large batch (4096+) to fill the negative pool, demanding multi-GPU with NCCL all-reduce.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

SimCLR, MoCo-v2/v3, BYOL, SimSiam, DINO, MAE (Masked Autoencoder), MoCo-CXR (chest X-ray adaptation), momentum encoder, projection head, cosine-similarity loss, online-vs-target network paradigm.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/self-supervised-image-pretraining-for-medical-imaging.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/self-supervised-image-pretraining-for-medical-imaging.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\self-supervised-image-pretraining-for-medical-imaging.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: ChestX-ray14 / NIH — 112k chest X-rays with 14 disease labels (https://nihcc.app.box.com/v/ChestXray-NIHCC) MIMIC-CXR — 227k X-rays for self-supervised pretraining (https://physionet.org/content/mimic-cxr/) RadImageNet — 1.35M radiology images across CT/MRI/US for SSL pretraining (verify URL) BraTS + TCIA — unlabelled MRI/CT volumes for 3D MAE pretraining (https://www.cancerimagingarchive.net/)

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

MoCo-CXR (https://arxiv.org/abs/2010.05352) — MoCo applied to chest X-ray, code available (verify URL) MONAI self-supervised (https://github.com/Project-MONAI/research-contributions) — MAE and contrastive pretraining for 3D medical images DINO (https://github.com/facebookresearch/dino) — self-supervised ViT; adaptable to medical imaging lightly (https://github.com/lightly-ai/lightly) — SSL framework supporting SimCLR/DINO/MoCo on GPU

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

NCCL all-reduce for multi-GPU negative queue synchronisation, cuDNN for backbone convolutions, cuBLAS for projection head; pattern: large-batch contrastive with NCCL gradient sync, or momentum queue stored in GPU SRAM. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
