# 7.15 — Uncertainty & Out-of-Distribution Detection in Medical AI

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Medical%20AI%20%26%20Clinical%20Deep%20Learning-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 7: Medical AI & Clinical Deep Learning · Catalog ID `7.15`
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

Quantifies model uncertainty for medical predictions to flag distribution shift (e.g., a new imaging protocol, rare pathology) and prevent silent failures in deployed AI. Bayesian approximations (MC Dropout, Deep Ensembles, SWAG) require multiple stochastic forward passes — naturally parallelised across ensemble members on GPU. Conformal prediction calibrates coverage guarantees without distribution assumptions, requiring GPU-parallelised score computation over large calibration sets. Energy-based OOD scoring on medical images computes a scalar energy per sample in a parallelised batch. The research challenge is calibrating uncertainty to actual clinical risk without access to labelled OOD data.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

MC Dropout (Gal & Ghahramani), Deep Ensembles, SWAG (SWA-Gaussian), Normalising Flows for density estimation, Energy-Based Models, Mahalanobis OOD detection, Conformal Prediction (split, cross-conformal), Temperature Scaling, Label Smoothing.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/uncertainty-out-of-distribution-detection-in-medical-ai.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/uncertainty-out-of-distribution-detection-in-medical-ai.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\uncertainty-out-of-distribution-detection-in-medical-ai.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Camelyon17 (WILDS) — histopathology with explicit hospital distribution shift (https://wilds.stanford.edu/datasets/#camelyon17) MIMIC-CXR — train/test splits across demographic strata for OOD evaluation (https://physionet.org/content/mimic-cxr/) RSNA Pneumonia Detection — Kaggle competition dataset for OOD robustness benchmarks (verify URL) MedMNIST — 18 standardised 2D/3D medical classification tasks for OOD benchmarking (https://medmnist.com/)

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

WILDS (https://github.com/p-lambda/wilds) — distribution shift benchmark with CausalML support PyTorch-Uncertainty (https://github.com/ENSTA-U2IS-AI/torch-uncertainty) — uncertainty methods on GPU ConformalCI (https://github.com/aangelopoulos/conformal-prediction) — conformal prediction calibration Laplace-Redux (https://github.com/AlexImmer/Laplace) — post-hoc Laplace approximation for pretrained NNs

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuDNN for parallel ensemble member forward passes, cuBLAS for Mahalanobis computation over feature covariance matrix; pattern: batch-parallel ensemble evaluation with stacked model weights. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
