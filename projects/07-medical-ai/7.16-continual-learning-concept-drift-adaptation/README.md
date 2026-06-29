# 7.16 — Continual Learning & Concept Drift Adaptation

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Medical%20AI%20%26%20Clinical%20Deep%20Learning-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 7: Medical AI & Clinical Deep Learning · Catalog ID `7.16`
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

Enables deployed medical AI models to continuously incorporate new clinical data (new patient cohorts, updated imaging protocols, population shifts) without catastrophic forgetting of previously learned tasks. Experience replay stores old data in a GPU-resident memory buffer; elastic weight consolidation (EWC) computes Fisher information diagonals — a batched gradient-squared operation on GPU. Gradient Episodic Memory (GEM) requires projecting gradients to the feasible cone defined by old-task gradients, a GPU-parallelised quadratic program. Healthcare settings impose strict constraints: models cannot forget rare disease patterns seen only in early training.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Elastic Weight Consolidation (EWC), Progressive Neural Networks, PackNet, Gradient Episodic Memory (GEM), Experience Replay (ER), Dark Experience Replay (DER++), Learning Without Forgetting (LwF), Online EWC.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/continual-learning-concept-drift-adaptation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/continual-learning-concept-drift-adaptation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\continual-learning-concept-drift-adaptation.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: MIMIC-IV — temporal partitioning by year to simulate concept drift (https://physionet.org/content/mimiciv/) CheXpert / MIMIC-CXR — multi-cohort splits for sequential task training (https://stanfordmlgroup.github.io/competitions/chexpert/) MedMNIST — 18-task sequential benchmark (https://medmnist.com/) Skin Lesion datasets (ISIC archive) — year-stratified splits for drift simulation (https://www.isic-archive.com/)

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

Avalanche (https://github.com/ContinualAI/avalanche) — continual learning library with GPU support and medical imaging plugins Mammoth (https://github.com/aimagelab/mammoth) — GPU continual learning framework with DER++, GEM, EWC FACIL (https://github.com/mmasana/FACIL) — class-incremental learning on GPU for image classifiers CLMNIST / MedicalCL (verify URL) — medical imaging continual learning benchmarks

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuBLAS for Fisher diagonal computation, CUDA replay buffer sampling with pinned memory; pattern: gradient projection via CUDA-parallelised QP over constraint matrices. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
