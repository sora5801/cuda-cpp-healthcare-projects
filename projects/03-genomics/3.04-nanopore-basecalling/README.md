# 3.4 — Nanopore Basecalling

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟢 Beginner · Established** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.4`
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

Nanopore basecalling translates raw ionic-current signal samples (electrical squiggles) from the sequencer into DNA/RNA base sequences. Oxford Nanopore's Dorado uses a recurrent neural network (transformer + CTC decoder in current "SUP" models) trained to map signal windows to base probabilities. The bottleneck is the RNN/transformer inference over millions of signal events per run hour, a perfect GPU workload: batched matrix multiplications across reads mapped to thousands of CUDA cores. Dorado achieves up to 30% speed improvement for HAC models on Ampere/Ada/Blackwell GPUs over previous versions and scales linearly across multiple GPUs. The GPU also powers modified base (methylation) calling simultaneously during basecalling.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Bidirectional LSTM / Transformer encoder; Connectionist Temporal Classification (CTC) decoding; beam search decoding; adaptive banded event alignment (f5c); Modified base (5mC, 6mA) classification heads.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/nanopore-basecalling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/nanopore-basecalling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\nanopore-basecalling.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: ONT Open Dataset (PromethION human WGS) — available via SRA / ENA (https://www.ncbi.nlm.nih.gov/sra); R9.4.1 and R10.4.1 benchmark datasets released by ONT (https://github.com/GoekeLab/awesome-nanopore); GIAB ONT ultra-long reads — NA12878/HG002 nanopore truth sets (https://www.nist.gov/programs-projects/genome-bottle); ENA Project PRJNA594038 — public multi-species ONT data (https://www.ebi.ac.uk/ena).

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

Dorado (https://github.com/nanoporetech/dorado) — ONT's official GPU basecaller, multi-GPU, CUDA-optimised, supports MOD calling; f5c (https://github.com/hasindu2008/f5c) — CUDA-accelerated methylation calling and event alignment; awesome-nanopore (https://github.com/GoekeLab/awesome-nanopore) — curated tool index including GPU-enabled callers; Guppy — legacy ONT CUDA basecaller, GPU-only, superseded by Dorado.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuDNN (RNN/transformer), TensorRT (inference optimisation), cuBLAS (GEMM), CUDA streams (pipelining signal batches); multi-GPU with NVLink/NCCL; persistent thread blocks for stateful RNN across signal chunks. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
