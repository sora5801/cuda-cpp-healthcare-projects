# 3.18 — Protein Language Model Inference

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.18`
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

Protein language models (PLMs) such as Meta's ESM-2 (650 M–15 B parameters) learn evolutionary constraints from hundreds of millions of protein sequences; their residue embeddings encode structure, function, and mutational effects. ESMFold uses ESM-2 as a trunk to predict 3D structure without MSA, making it dramatically faster than AlphaFold2 for single-sequence predictions. GPU acceleration of the multi-head self-attention layers (O(L²) per layer for sequence length L) is essential—H100 Tensor Cores achieve >3× MFU for these GEMM workloads. Inference of 10 M UniProt proteins via ESMFold required a dedicated GPU cluster; GPU batching of mixed-length proteins with padding optimisation is the key engineering challenge.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Transformer multi-head self-attention (Q×K^T scaling, softmax, V aggregation); rotary positional embeddings; evoformer-style structure module; invariant point attention (IPA); masked language model (MLM) training; FlashAttention memory-efficient attention.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/protein-language-model-inference.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/protein-language-model-inference.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\protein-language-model-inference.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: UniRef50/90 — training corpus for PLMs (https://www.uniprot.org/help/uniref); ESM Metagenomic Atlas — 700 M metagenomic protein structures (https://esmatlas.com/); PDB structures — validation set for ESMFold (https://www.rcsb.org/); CATH / SCOP — structural classification databases (https://www.cathdb.info/).

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

fair-esm (https://github.com/facebookresearch/esm) — Meta's ESM-2 and ESMFold, official CUDA inference code; EvolutionaryScale ESM3 (https://github.com/evolutionaryscale/esm) — latest multimodal protein model; ColabFold (https://github.com/sokrypton/ColabFold) — fast MSA + AlphaFold2 on GPU; xTrimoPGLM (https://huggingface.co/BonjwrAI/xTrimoPGLM-100B) — 100 B protein LM (verify URL).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuDNN / Apex / FlashAttention-2 for attention; cuBLAS GEMM for feed-forward layers; Tensor Core FP16/BF16 mixed precision; multi-GPU tensor + pipeline parallelism (Megatron-LM / DeepSpeed); dynamic batching by sequence length bucket. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
