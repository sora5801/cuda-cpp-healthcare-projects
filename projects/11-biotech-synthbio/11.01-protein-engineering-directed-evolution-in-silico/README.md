# 11.1 — Protein Engineering / Directed Evolution In Silico

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Biotechnology%2C%20Bioprocess%20%26%20Synthetic%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 11: Biotechnology, Bioprocess & Synthetic Biology · Catalog ID `11.1`
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

Machine-learning-guided directed evolution replaces physical screening with GPU-accelerated fitness prediction, scoring millions of sequence variants per second using protein language models (ESM-2) or structure-based Rosetta energy functions. EVOLVEpro (Science 2025) demonstrated rapid in silico directed evolution by proposing and filtering variants with GPU-deployed LLM embeddings. Batched GPU inference over combinatorial mutation libraries (10⁸–10¹² sequences) identifies beneficial mutations orders of magnitude faster than laboratory selection. The key parallelism is embarrassingly parallel: each sequence variant scores independently.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Protein language model (ESM-2) embeddings + fitness regression, directed evolution with Bayesian optimization (GP or Bayesian neural network), structure-based ΔΔG prediction (Rosetta fast-relax, FoldX), zero-shot fitness scoring via masked-language-model log-odds, gradient-based sequence optimization via differentiable fitness surrogate.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/protein-engineering-directed-evolution-in-silico.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/protein-engineering-directed-evolution-in-silico.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\protein-engineering-directed-evolution-in-silico.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: ProteinGym Substitution Benchmarks — 250+ deep mutational scanning (DMS) datasets across protein families (https://proteingym.org/); Envision (PABP, UBE4B) fitness landscapes; Fluorescent Protein Dataset (GFP) — 56 K variants with fluorescence labels (https://github.com/fhalab/FLIP); FLIP Benchmarks — standardized fitness landscape benchmarks (https://github.com/J-SNACKKB/FLIP).

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

ESM (https://github.com/facebookresearch/esm) — Meta FAIR ESM-2 + ESMFold GPU protein language model; EVOLVEpro (verify URL at bakerlab.org or GitHub) — in silico directed evolution pipeline; ProteinMPNN (https://github.com/dauparas/ProteinMPNN) — GPU sequence design from backbone; Fitness-Prediction-Benchmark (https://github.com/J-SNACKKB/FLIP) — DMS benchmark datasets and baseline models.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuDNN for Transformer forward pass over batched sequences, Flash Attention for memory-efficient long-sequence attention, mixed-precision (BF16) for throughput; pattern: encode 10⁶ variants as token batch → GPU LLM forward pass → fitness score vector → Bayesian acquisition function selects next round → iterate. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
