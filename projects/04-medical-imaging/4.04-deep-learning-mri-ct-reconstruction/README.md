# 4.4 — Deep-Learning MRI/CT Reconstruction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.4`
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

Learned reconstruction networks replace hand-crafted priors with data-driven mappings from under-sampled/degraded k-space or sinogram to fully-sampled images. End-to-end variational networks (E2E-VarNet) unroll gradient descent iterations as network layers, each with trainable sensitivity maps and refinement modules; these run entirely on GPU during both training (batch gradient descent) and inference. Training on large multi-coil raw k-space datasets (fastMRI) requires TB-scale data loading with GPU-pinned memory and mixed-precision FP16/BF16 tensor cores. Inference at 256² × 32 coils can achieve sub-100 ms per volume on a single A100, enabling real-time clinical deployment.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

E2E-VarNet (variational network with learned sensitivity maps), unrolled ADMM-Net, deep cascade of CNN, U-Net in image domain, score-based diffusion models for MRI (DiffusionMBIR), plug-and-play denoising priors, recurrent unrolled networks.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/deep-learning-mri-ct-reconstruction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/deep-learning-mri-ct-reconstruction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\deep-learning-mri-ct-reconstruction.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: fastMRI (https://fastmri.med.nyu.edu/) — raw multi-coil k-space, knee/brain, gold-standard reference; fastMRI+ with radiologist annotations (https://github.com/StanfordMIMI/fastMRI_plus); 2016 AAPM Low-Dose CT Challenge for CT reconstruction learning.

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

fastMRI baseline code (https://github.com/facebookresearch/fastMRI) — PyTorch E2E-VarNet, U-Net, evaluation scripts; BART (https://github.com/mrirecon/bart) — Deep MRI reconstruction via BART-learn module; Direct (https://github.com/directgroup/direct) — modular PyTorch framework for DL MRI reconstruction (multiple unrolled architectures); Hugging Face Medical Imaging (https://huggingface.co/datasets?search=mri) — model hub with pretrained MRI reconstruction checkpoints.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuDNN (conv layers), Tensor Cores (FP16 mixed precision), PyTorch CUDA autograd; pipeline: data → pinned host memory → GPU → network forward pass → loss → backward; multi-GPU DDP training via NCCL. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
