# 12.6 — Microbiome & Antimicrobial-Resistance Analytics

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Analytical%20%26%20Omics%20Data%20Processing-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 12: Analytical & Omics Data Processing · Catalog ID `12.6`
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

Microbiome profiling from shotgun metagenomics combines taxonomic classification (GPU Kraken2 / MetaCache) with functional annotation (GPU DIAMOND / MMseqs2 vs. CARD/ResFinder for AMR genes) and community ecology statistics. The AMR gene identification step—aligning millions of reads against thousands of resistance gene models (RGI uses DIAMOND + CARD)—is the most GPU-amenable component. Deep learning models (MSDeepAMR, DeepARG) trained on genomic features or mass spectrometry (MALDI-TOF) patterns predict resistance phenotypes and are accelerated by GPU inference. Metagenome-assembled genome (MAG) binning via deep learning (DAS_Tool) is also GPU-amenable.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

K-mer-based taxonomic classification (Kraken2/MetaCache); protein homology search vs. AMR databases (DIAMOND/CARD); profile HMM search for resistance gene families; MALDI-TOF spectral CNN for phenotypic AMR prediction; random forest / gradient boosting for AMR genotype-to-phenotype; deep learning MAG binning.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/microbiome-antimicrobial-resistance-analytics.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/microbiome-antimicrobial-resistance-analytics.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\microbiome-antimicrobial-resistance-analytics.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: CARD — Comprehensive Antibiotic Resistance Database (https://card.mcmaster.ca/); PATRIC / BV-BRC — bacterial pathogen genomes (https://www.bv-brc.org/); CAMDA AMR challenge datasets (http://www.camda.info/); HMP2 (Human Microbiome Project Phase 2) (https://www.hmpdacc.org/).

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

MetaCache-GPU (https://arxiv.org/pdf/2106.08150) — GPU metagenomic classifier; DIAMOND (https://github.com/bbuchfink/diamond) — GPU-targetable protein aligner for AMR annotation; DeepARG (https://github.com/gaarangoa/deeparg) — deep learning AMR gene predictor (GPU inference); RGI (https://github.com/arpcard/rgi) — Resistance Gene Identifier using CARD database.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

GPU hash tables for k-mer AMR classification; batched cuDNN CNN inference for MALDI spectral AMR prediction; cuBLAS for alignment scoring matrix; thrust for read partition by taxon; RAPIDS cuDF for large microbiome count matrix operations. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
