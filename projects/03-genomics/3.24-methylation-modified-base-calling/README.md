# 3.24 — Methylation / Modified-Base Calling

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.24`
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

Detection of DNA methylation (5mC, 5hmC) and other modifications (6mA, BrdU) from nanopore raw signal requires classifying the ionic current waveform at each potentially modified site. f5c's GPU-accelerated adaptive banded event alignment assigns signal events to reference positions using GPU-parallelised DP, then scores modification probability. ONT Remora trains small CNN/LSTM models to classify modifications directly from raw signals, with GPU inference integrated into Dorado basecalling. Galaxy-methyl achieves 3–5× GPU speedup over f5c via parallelised methylation score kernels. Accurate genome-wide 5mCG calling at 30× ONT coverage processes billions of signal samples.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Adaptive banded event-alignment DP (f5c); CTC basecalling with modification-aware output alphabet (Dorado/Remora); CNN/LSTM classification of signal windows per site; log-likelihood ratio modification scoring; binomial model for allele-specific methylation; bisulfite-seq Viterbi (for BS-seq comparison).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/methylation-modified-base-calling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/methylation-modified-base-calling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\methylation-modified-base-calling.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: ENCODE WGBS — genome-wide bisulfite methylation reference (https://www.encodeproject.org/); Oxford Nanopore open datasets — R10.4.1 with 5mC/6mA labels (https://github.com/GoekeLab/awesome-nanopore); NCBI GEO methylation studies (https://www.ncbi.nlm.nih.gov/geo/); ENCODE long-read methylation data (https://www.encodeproject.org/).

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

f5c (https://github.com/hasindu2008/f5c) — CUDA-accelerated methylation calling and event alignment; Remora (https://github.com/nanoporetech/remora) — ONT modified base model training and calling; Dorado (https://github.com/nanoporetech/dorado) — integrates modification calling during basecalling on GPU; Modkit (https://github.com/nanoporetech/modkit) — modified base analysis downstream of Dorado.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Adaptive banded DP in CUDA shared memory; cuDNN for RNN/CNN modification classifier; persistent threads for streaming signal batches; CUDA streams for multi-read GPU pipeline; warp-level primitives for log-likelihood reduction. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
