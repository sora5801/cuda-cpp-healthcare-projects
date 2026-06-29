# 8.7 — EEG Seizure Detection & Prediction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Neuroscience%20%26%20Brain--Computer%20Interfaces-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 8: Neuroscience & Brain-Computer Interfaces · Catalog ID `8.7`
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

Epileptic seizure prediction from scalp EEG requires continuous multi-channel spectral feature extraction and classification over rolling windows with latencies <1 s. The preictal period (minutes to hours before seizure onset) exhibits subtle changes in high-frequency oscillations (HFOs), phase-amplitude coupling, and cross-channel coherence. GPU allows real-time feature extraction from 256 channels × 2 500 Hz using cuFFT spectrograms, simultaneous CNN/LSTM classification, and sliding-window cross-correlation for connectivity graphs.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Short-time Fourier transform (STFT), Morlet wavelet, phase-amplitude coupling (PAC), graph-theoretic seizure propagation, 1D-CNN and BiLSTM classifiers, attention transformer for long-range EEG context, support vector machine (SVM) on spectral features, SEEG source imaging.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/eeg-seizure-detection-prediction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/eeg-seizure-detection-prediction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\eeg-seizure-detection-prediction.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Temple University Hospital EEG Corpus (TUAB/TUEG) — 30 000+ EEG recordings (https://isip.piconepress.com/projects/tuh_eeg/); CHB-MIT Scalp EEG Database (PhysioNet) (https://physionet.org/content/chbmit/1.0.0/); IEEG Portal — intracranial EEG for epilepsy (https://www.ieeg.org); OpenNeuro epilepsy datasets (https://openneuro.org).

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

MNE-Python (https://github.com/mne-tools/mne-python) — EEG processing with parallel backend; PyTorch EEG (https://github.com/torcheeg/torcheeg) — GPU deep learning for EEG; EEGLAB (https://github.com/sccn/eeglab) — MATLAB seizure analysis plugins; BrainFlow (https://github.com/brainflow-dev/brainflow) — real-time streaming for wearable seizure monitors.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuFFT batched STFT across all channels simultaneously; cuDNN for CNN classifier inference; custom CUDA kernel for phase-amplitude coupling across channel pairs; pattern: rolling window with circular GPU buffer, cuFFT on each frame, classifier inference on extracted features via TensorRT. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
