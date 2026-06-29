# 10.6 — Wearable & Continuous-Sensor Signal Processing

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Biomechanics%2C%20Biomedical%20Devices%20%26%20Surgery-lightgrey)

> **🟢 Beginner · Established** — Domain 10: Biomechanics, Biomedical Devices & Surgery · Catalog ID `10.6`
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

Wearable ECG, EMG, EEG, PPG, and IMU streams generate high-throughput multi-channel time-series that require real-time filtering, feature extraction, and classification. GPU acceleration enables sliding-window FFT across 64+ channels simultaneously, convolution-based band-pass filtering in the frequency domain, and inference of deep neural networks (CNN-LSTM, Transformer) on segmented epochs. Continuous-glucose-monitor (CGM) and wearable ECG (Holter) datasets reach billions of samples per patient-day, requiring GPU-accelerated dynamic time warping, anomaly detection, and arrhythmia classification pipelines. Edge GPU (Jetson Orin) deployments compress and quantize models for on-device inference.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Short-time Fourier transform (STFT), wavelet packet decomposition, matched-filter arrhythmia detection, CNN-LSTM for HAR (human activity recognition), dynamic time warping (DTW), Kalman/Madgwick filter for IMU fusion, federated learning over distributed wearables.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/wearable-continuous-sensor-signal-processing.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/wearable-continuous-sensor-signal-processing.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\wearable-continuous-sensor-signal-processing.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: PhysioNet/CinC Challenge — ECG arrhythmia (https://physionet.org/); PAMAP2 Physical Activity Monitoring — IMU + heart rate across 18 activities (https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring); MIT-BIH Arrhythmia Database — annotated 2-channel ECG (https://physionet.org/content/mitdb/1.0.0/); CHB-MIT Scalp EEG — epileptic seizure monitoring (https://physionet.org/content/chbmit/1.0.0/).

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

cuSignal (https://github.com/rapidsai/cusignal) — RAPIDS GPU signal processing library (drop-in scipy.signal on GPU); NeuroKit2 (https://github.com/neuropsychology/NeuroKit) — biosignal processing (CPU; GPU backend extensible); TorchEEG (https://github.com/torcheeg/torcheeg) — GPU-accelerated EEG deep-learning benchmark framework; PhysioNet WFDB Python (https://github.com/MIT-LCP/wfdb-python) — waveform database I/O.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuFFT (batch FFT over channels), cuDNN (CNN/LSTM inference), CUDA kernels for sliding-window feature extraction; pattern: ring-buffer ingest → batch cuFFT per window → 1D convolution via cuDNN → softmax classification → alert emission with sub-10 ms latency. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
