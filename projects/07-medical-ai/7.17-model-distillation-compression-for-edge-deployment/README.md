# 7.17 — Model Distillation & Compression for Edge Deployment

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20AI%20%26%20Clinical%20Deep%20Learning-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 7: Medical AI & Clinical Deep Learning · Catalog ID `7.17`
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

Compresses large clinical AI models (ViT-L, GPT-style) into small student networks that fit on embedded devices by matching teacher logits, intermediate features, or attention maps. Knowledge distillation training is GPU-bound: both teacher and student run forward passes in every iteration, doubling compute vs. standard training. Structured pruning removes entire channels or attention heads; the resulting sparse model still benefits from GPU execution through efficient sparse tensor routines. INT8 quantisation-aware training (QAT) uses fake-quantisation operators that are CUDA-kernel-friendly and allow recovery of accuracy on medical benchmarks.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Soft label distillation (Hinton et al.), feature matching (FitNets), attention transfer, data-free distillation, structured/unstructured pruning, magnitude-based weight pruning, quantisation-aware training (QAT), GPTQ, AWQ, LoRA for student warm-start.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/model-distillation-compression-for-edge-deployment.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/model-distillation-compression-for-edge-deployment.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\model-distillation-compression-for-edge-deployment.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: ImageNet (pre-training teachers) + CheXpert (domain fine-tuning) — dual-dataset compression pipeline MedMNIST — small-scale medical benchmark for student evaluation (https://medmnist.com/) PTB-XL — ECG dataset for waveform model compression evaluation (https://physionet.org/content/ptb-xl/) LIDC-IDRI — CT nodule dataset for compressed segmentation model evaluation (https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)

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

TensorRT (https://github.com/NVIDIA/TensorRT) — quantisation and layer fusion for medical inference Intel Neural Compressor (https://github.com/intel/neural-compressor) — INT8 QAT and pruning on GPU/CPU Pytorch-Distiller (verify URL) — knowledge distillation toolkit Once-for-All (https://github.com/mit-han-lab/once-for-all) — NAS + distillation for efficient medical model families

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuDNN for both teacher/student simultaneous forward passes, TensorRT PTQ/QAT calibration; pattern: data-parallel joint teacher-student training with teacher frozen on GPU. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
