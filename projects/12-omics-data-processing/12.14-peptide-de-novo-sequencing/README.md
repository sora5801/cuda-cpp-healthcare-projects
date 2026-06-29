# 12.14 — Peptide De Novo Sequencing

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Analytical%20%26%20Omics%20Data%20Processing-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 12: Analytical & Omics Data Processing · Catalog ID `12.14`
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

De novo peptide sequencing infers amino acid sequences directly from MS/MS spectra without a protein database, critical for non-model organisms, immunopeptidomics, and modified peptides. Algorithms generate candidate sequences by traversing a spectrum graph (nodes = fragment ions, edges = amino acid mass differences) via beam search or dynamic programming. GPU acceleration applies to: (1) GPU-parallel beam search over thousands of candidate sequences simultaneously, (2) batched transformer/LSTM scoring of candidate sequences, and (3) the CUDA-accelerated knapsack DP ensuring precursor mass consistency. NovoBench (NeurIPS 2024) benchmarks GPU-accelerated deep learning de novo sequencers.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Spectrum graph construction (b/y-ion nodes); beam-search decoding with GPU-parallel branches; CUDA knapsack DP for precursor mass constraint; seq2seq transformer (Casanovo, PointNovo); bidirectional LSTM encoder; attention over fragment ion sequence; PTM-tolerant open search.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/peptide-de-novo-sequencing.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/peptide-de-novo-sequencing.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\peptide-de-novo-sequencing.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: PRIDE ProteomeXchange benchmark de novo datasets (https://www.ebi.ac.uk/pride/); NovoBench benchmark (https://github.com/jingbo02/NovoBench) — standardised deep learning de novo benchmark; MassIVE (https://massive.ucsd.edu/); PeptideAtlas synthetic peptide datasets (https://www.peptideatlas.org/).

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

Casanovo (https://github.com/Noble-Lab/casanovo) — transformer-based GPU de novo sequencer; NovoBench (https://github.com/jingbo02/NovoBench) — NeurIPS 2024 benchmark suite; PointNovo (verify URL, from Ma et al.) — deep learning de novo with GPU inference; DeepNovo (https://github.com/nh2tran/DeepNovo) — original LSTM-based GPU de novo sequencer.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuDNN transformer/LSTM inference; CUDA knapsack DP (shared-memory DP table per spectrum); batched beam search with GPU-parallel candidate scoring; Tensor Core BF16 for transformer scoring; one CUDA stream per spectrum batch. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
