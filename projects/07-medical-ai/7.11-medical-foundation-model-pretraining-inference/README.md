# 7.11 — Medical Foundation-Model Pretraining & Inference

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20AI%20%26%20Clinical%20Deep%20Learning-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 7: Medical AI & Clinical Deep Learning · Catalog ID `7.11`
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

Pretrains large-scale (1B–70B parameter) language, vision, or multimodal models on domain-specific medical corpora — PubMed, MIMIC clinical notes, radiology report databases, pathology image collections — to produce general-purpose medical representations. Pretraining is massively GPU-bound: the matrix multiplications in transformer attention and feed-forward layers constitute >95% of FLOPs. Tensor-parallel and pipeline-parallel model partitioning across hundreds of A100/H100 GPUs (via Megatron-LM or DeepSpeed) is necessary for 70B-parameter models. Inference serving uses Flash Attention, continuous batching (vLLM), and INT8/GPTQ quantisation to handle concurrent clinical queries.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Autoregressive pretraining (GPT), masked language modelling (BERT), instruction tuning (SFT + RLHF), Vision-Language Contrastive pretraining (CLIP, FLAVA), Mixture-of-Experts (MoE), FlashAttention-2, LoRA/QLoRA fine-tuning, GPTQ quantisation.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/medical-foundation-model-pretraining-inference.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/medical-foundation-model-pretraining-inference.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\medical-foundation-model-pretraining-inference.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: PubMed Central Open Access — 4M+ full biomedical articles (https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/) MIMIC-IV Notes — 331,794 clinical notes (https://physionet.org/content/mimic-iv-note/) The Pile: Pile-MedMent / S2ORC — broad scientific pretraining corpora (https://pile.eleuther.ai/) OpenPath / PathCap — pathology image-caption pairs for vision-language pretraining (verify URL)

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

MEDITRON (https://github.com/epfLLM/meditron) — Llama-2 70B adapted for medicine with GPU pretraining scripts Awesome Healthcare Foundation Models (https://github.com/Jianing-Qiu/Awesome-Healthcare-Foundation-Models) — curated model list Awesome Foundation Models in Medical Imaging (https://github.com/xmindflow/Awesome-Foundation-Models-in-Medical-Imaging) — curated vision-language models vLLM (https://github.com/vllm-project/vllm) — continuous batching inference engine for serving medical LLMs on GPU

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Megatron-LM tensor parallelism, DeepSpeed ZeRO, Flash Attention 2, NCCL all-reduce; pattern: 3D parallelism (tensor × pipeline × data), NVLink high-bandwidth GPU fabric required. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
