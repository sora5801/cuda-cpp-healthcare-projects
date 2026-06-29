# 7.12 — Sepsis Early Warning System

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20AI%20%26%20Clinical%20Deep%20Learning-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 7: Medical AI & Clinical Deep Learning · Catalog ID `7.12`
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

Predicts the onset of sepsis 3–6 hours before clinical recognition from streaming ICU vitals, lab values, and medication records using recurrent or transformer architectures. The GPU bottleneck is batched forward passes through temporal models (LSTM, GRU, Transformer-XL) over thousands of patient time series simultaneously. Real-time deployment requires sub-second latency over continuously appended EHR streams. Processing irregular time-series (lab values arrive at non-uniform intervals) requires attention mechanisms that weigh observations by recency and relevance — these attention operations are CUDA-accelerated. Large training cohorts (>100k ICU admissions) sustain continuous GPU utilisation throughout training.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

LSTM/GRU temporal classifiers, Transformer-XL for long EHR sequences, Temporal Fusion Transformers (TFT), missing-value imputation via learned decay, AUROC-calibrated threshold selection, early stopping with Clinical Early Warning Scores (qSOFA, SOFA) as baselines, conformal prediction for uncertainty.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/sepsis-early-warning-system.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/sepsis-early-warning-system.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\sepsis-early-warning-system.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: MIMIC-Sepsis benchmark (https://arxiv.org/abs/2510.24500) — curated sepsis trajectory subset of MIMIC-IV eICU-CRD — 200k+ admissions, multi-site for generalisation testing (https://eicu-crd.mit.edu/) PhysioNet/Computing in Cardiology Challenge 2019 — sepsis prediction from ICU time series (https://physionet.org/content/challenge-2019/) HiRID — high-resolution ICU dataset from Bern University Hospital (https://physionet.org/content/hirid/)

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

MIMIC-Extract (https://github.com/MLforHealth/MIMIC_Extract) — standardised MIMIC ICU feature tables PyHealth (https://github.com/sunlabuiuc/PyHealth) — healthcare AI library with ICU prediction tasks on GPU ETHOS (verify URL) — transformer-based sepsis prediction on EHR tokens Temporal Fusion Transformer (https://github.com/jdb78/pytorch-forecasting) — multi-horizon temporal model with GPU support

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuDNN for LSTM/GRU cells, Flash Attention for transformer EHR models, Thrust for sorting irregular timestamps; pattern: padded minibatch of patient time series with masking, GPU-resident rolling window inference for real-time alerting. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
