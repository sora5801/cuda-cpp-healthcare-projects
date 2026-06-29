# 7.5 — Federated Learning for Healthcare

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20AI%20%26%20Clinical%20Deep%20Learning-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 7: Medical AI & Clinical Deep Learning · Catalog ID `7.5`
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

Trains a single global model across multiple hospitals without sharing raw patient data: each site trains on local data and sends only model gradients or weights to a central aggregator. The GPU bottleneck on each client is identical to standard local training; additional communication cost arises from the aggregation step. NVIDIA FLARE orchestrates GPU-based local training with differential privacy noise injection and secure aggregation. Heterogeneous GPU fleets across hospitals (V100 at one site, A100 at another) require adaptive batch sizing and mixed-precision logic. The primary research challenge is handling statistical data heterogeneity (non-IID distributions) while maintaining convergence.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

FedAvg, FedProx, SCAFFOLD, FedNova, personalised federated learning, differential privacy (Gaussian mechanism, moments accountant), secure aggregation with homomorphic encryption, communication compression (gradient sparsification, quantisation).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/federated-learning-for-healthcare.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/federated-learning-for-healthcare.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\federated-learning-for-healthcare.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: TCGA (The Cancer Genome Atlas) — multi-institutional genomics + histopathology (https://www.cancer.gov/tcga) MIMIC-IV — EHR data used in federated simulation across synthetic partitions (https://physionet.org/content/mimiciv/) NIH Chest X-ray Dataset — 112,120 chest X-rays for FL benchmarks (https://nihcc.app.box.com/v/ChestXray-NIHCC) Medical Segmentation Decathlon — multi-task dataset used in FL challenges (http://medicaldecathlon.com/)

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

NVIDIA FLARE (https://github.com/NVIDIA/NVFlare) — production-grade federated learning SDK with GPU-native training loops OpenFL (https://github.com/securefederatedai/openfl) — Intel/Linux Foundation FL framework supporting PyTorch/TF on GPU Flower (https://github.com/adap/flower) — lightweight, framework-agnostic FL with GPU support PySyft (https://github.com/OpenMined/PySyft) — privacy-preserving FL with differential privacy on GPU

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuDNN for local model training, NCCL for efficient intra-site multi-GPU; pattern: data parallelism within site, synchronous or asynchronous gradient aggregation between sites via secure channels. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
