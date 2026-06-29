# 7.18 — Retinal Fundus AI Screening

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Medical%20AI%20%26%20Clinical%20Deep%20Learning-lightgrey)

> **🟢 Beginner · Established** — Domain 7: Medical AI & Clinical Deep Learning · Catalog ID `7.18`
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

Classifies diabetic retinopathy, glaucoma, and age-related macular degeneration from colour fundus photographs or OCT scans. High-resolution fundus images (typically 2048×2048) require significant GPU memory for batch processing; ResNet, EfficientNet, and Swin-Transformer backbones fine-tuned on annotated fundus datasets are the standard approach. GPU tensor cores accelerate the backbone convolutions in batch; simultaneous inference across both eyes and multiple pathologies (multi-task heads) doubles effective throughput. Real-world screening pipelines process millions of images annually, making GPU throughput a primary operational concern.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

EfficientNet-B4/B5 (winner of EyePACS 2019), Swin Transformer, Grad-CAM for lesion localisation, multi-task classification (DR grade + glaucoma + AMD), self-supervised pretraining on unlabelled fundus images, uncertainty calibration for referral decisions.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/retinal-fundus-ai-screening.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/retinal-fundus-ai-screening.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\retinal-fundus-ai-screening.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: EyePACS — 88,000 labelled fundus images, 5-grade DR severity (Kaggle, verify URL) APTOS 2019 — 3,662 fundus images, DR grading competition (Kaggle, verify URL) DRIVE / STARE — retinal vessel segmentation datasets (verify URL) UK Biobank Retinal Imaging — 68k retinal fundus images with linked health records (https://www.ukbiobank.ac.uk/)

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

EfficientDet / EfficientNet (https://github.com/google/automl/tree/master/efficientnet) — strong fundus baselines MONAI (https://github.com/Project-MONAI/MONAI) — fundus classification pipelines RETFound (https://github.com/rmaphoh/RETFound_MAE) — MAE-pretrained retinal foundation model on 1.6M fundus images DeepDR Plus (verify URL) — end-to-end diabetic retinopathy screening system

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuDNN for EfficientNet/Swin convolutions, TensorRT for clinic deployment; pattern: data-parallel fine-tuning on high-resolution fundus batches with gradient accumulation. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
