# 8.3 — EEG/MEG Source Localization & Processing

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Neuroscience%20%26%20Brain--Computer%20Interfaces-lightgrey)

> **🟢 Beginner · Established** — Domain 8: Neuroscience & Brain-Computer Interfaces · Catalog ID `8.3`
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

EEG/MEG source localization solves the ill-posed inverse problem of estimating the distribution of neural current sources inside the brain from measurements at 64–306 scalp/sensor locations. Forward model computation (leadfield matrix) via BEM/FEM over a realistic head model is a one-time GPU-amenable precomputation. Inverse methods range from beamforming (spatial filtering) to sparse Bayesian learning (Champagne, SESAME) with large-scale matrix factorizations that benefit from GPU. Real-time EEG filtering for BCI or epilepsy monitoring requires FIR/IIR at 1 000–10 000 Hz on 256 channels.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Boundary element method (BEM) for leadfield computation, minimum norm estimate (MNE), LORETA / eLORETA, beamforming (LCMV, DICS), sparse Bayesian learning, MUSIC dipole scan, dynamical statistical parametric mapping (dSPM), time-frequency analysis (Morlet wavelet, multitaper).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/eeg-meg-source-localization-processing.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/eeg-meg-source-localization-processing.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\eeg-meg-source-localization-processing.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: OpenNeuro EEG/MEG datasets in BIDS (https://openneuro.org); DANDI neurophysiology archive (https://dandiarchive.org); Human Connectome Project MEG (https://db.humanconnectome.org); TUAB / TUEG Temple University Hospital EEG corpus (https://isip.piconepress.com/projects/tuh_eeg/).

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

MNE-Python (https://github.com/mne-tools/mne-python) — comprehensive EEG/MEG analysis with GPU-accelerated backends; FieldTrip (https://github.com/fieldtrip/fieldtrip) — MATLAB MEG/EEG toolbox with parallel toolbox support; Brainstorm (https://github.com/brainstorm-users/brainstorm) — GUI EEG/MEG analysis; EEGLAB (https://github.com/sccn/eeglab) — MATLAB plugin ecosystem for EEG.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuBLAS DGEMM for leadfield matrix multiply and beamformer weight computation; cuSOLVER for minimum-norm pseudoinverse; cuFFT for spectral analysis (all channels simultaneously); pattern: channel × time matrix operations on GPU, batch FFT across all channel pairs for coherence analysis. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
