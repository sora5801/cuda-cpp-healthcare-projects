# 7.9 — Real-Time Edge Inference for Medical Devices

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20AI%20%26%20Clinical%20Deep%20Learning-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 7: Medical AI & Clinical Deep Learning · Catalog ID `7.9`
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

Deploys neural networks on embedded GPUs (NVIDIA Jetson, AMD Versal) or medical device SoCs for real-time inference — ECG arrhythmia detection, pulse oximetry anomaly, surgical robot vision, ultrasound B-mode AI. The challenge is matching model latency to physiological sampling rates (e.g., 500 Hz ECG requires <2 ms inference). TensorRT INT8 quantisation reduces model size 4× with minimal accuracy loss. Layer fusion fuses sequential convolutions, activations, and normalisation into single CUDA kernels, eliminating memory bandwidth bottlenecks. NVIDIA Jetson Orin delivers 275 TOPS at 60 W, enabling full clinical-grade CNN inference locally without cloud round-trips.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Post-training quantisation (INT8, INT4), knowledge distillation, neural architecture search for edge (MobileNetV3, EfficientDet-Lite), layer fusion, structured pruning, TensorRT engine optimisation, latency-aware NAS.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/real-time-edge-inference-for-medical-devices.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/real-time-edge-inference-for-medical-devices.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\real-time-edge-inference-for-medical-devices.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: PhysioNet Challenge datasets — ECG, SpO2, EEG for device validation (https://physionet.org/) CAMUS cardiac ultrasound segmentation dataset (https://www.creatis.insa-lyon.fr/Challenge/camus/) EyePACS retinal fundus — used for on-device DR screening validation (verify URL) MIMIC-III Waveform Database — high-freq bedside monitor signals (https://physionet.org/content/mimicdb/)

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

TensorRT (https://github.com/NVIDIA/TensorRT) — inference optimisation with INT8 calibration and layer fusion NVIDIA Jetson SDK (https://developer.nvidia.com/embedded/jetpack) — Jetson-optimised libraries for edge GPU inference MONAI Deploy (https://github.com/Project-MONAI/monai-deploy) — clinical AI deployment framework with TensorRT backend OpenVINO (https://github.com/openvinotoolkit/openvino) — Intel edge inference toolkit for x86+iGPU devices

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

TensorRT INT8 for quantised inference, cuDNN for on-device convolutions, Triton Inference Server for multi-model serving; pattern: streaming inference pipeline with zero-copy pinned memory between sensor DMA and GPU. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
