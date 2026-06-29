# 7.10 — Physiological Signal & Waveform Analysis

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20AI%20%26%20Clinical%20Deep%20Learning-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 7: Medical AI & Clinical Deep Learning · Catalog ID `7.10`
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

Processes continuous high-frequency physiological waveforms — ECG (500–2000 Hz), EEG (256–2048 Hz), arterial blood pressure, photoplethysmography — for automated diagnosis, anomaly detection, and prognostication. Long waveform segments (minutes to hours) require 1D temporal convolutions or transformer attention over thousands of time steps; both operations are GPU-bound. Processing multi-lead ECG simultaneously (12 leads × 5000 samples) as a 2D image enables CNN classification with no waveform-specific code. Batch processing of thousands of 24-hour Holter monitors in parallel on GPU is the primary throughput bottleneck in clinical annotation pipelines.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

1D ResNet / Inception, temporal convolutional networks (TCN), WaveNet, Bidirectional LSTM, self-supervised waveform pretraining (wav2vec 2.0 for ECG), Short-Time Fourier Transform (STFT) + CNN, multi-scale attention, event detection with anchor-free detection heads.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/physiological-signal-waveform-analysis.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/physiological-signal-waveform-analysis.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\physiological-signal-waveform-analysis.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: PhysioNet Computing in Cardiology Challenge 2021 — 12-lead ECG from multiple cohorts (https://physionet.org/content/challenge-2021/) MIMIC-IV-ECG — 800k+ ECGs from MIMIC patients (https://physionet.org/content/mimic-iv-ecg/) PTB-XL — 21,837 12-lead ECGs with cardiologist labels (https://physionet.org/content/ptb-xl/) Temple University EEG Corpus (TUEG) — 20k+ hours of clinical EEG (https://isip.piconepress.com/projects/tuh_eeg/)

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

ECG-FM (https://github.com/bowang-lab/ecg-fm) — wav2vec-based ECG foundation model, 90M params, GPU-pretrained ESI (https://github.com/comp-well-org/ESI) — multimodal ECG + text contrastive pretraining foundation model CLEF ECG (https://github.com/Nokia-Bell-Labs/ecg-foundation-model) — single-lead ECG foundation model pretrained on 161k MIMIC patients MNE-Python (https://github.com/mne-tools/mne-python) — EEG/MEG processing; GPU via deep learning backends

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuFFT for Fourier-domain convolutions on waveforms, cuDNN for 1D temporal convolutions, Flash Attention for long-sequence transformers; pattern: data-parallel batch processing across thousands of waveform windows, streaming input pipeline from waveform database. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
