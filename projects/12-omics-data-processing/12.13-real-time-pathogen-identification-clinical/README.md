# 12.13 — Real-Time Pathogen Identification (Clinical)

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Analytical%20%26%20Omics%20Data%20Processing-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 12: Analytical & Omics Data Processing · Catalog ID `12.13`
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

Clinical metagenomic next-generation sequencing (mNGS) for pathogen identification requires processing millions of reads within 1–2 hours of sample collection to guide antibiotic therapy. The critical path is: GPU basecalling (Dorado) → GPU k-mer classification (MetaCache/GPU Kraken2) → GPU AMR gene annotation (DIAMOND vs. CARD) → statistical confidence scoring. A 2024 MDPI paper describes a GPU-integrated nanopore workstation running CUDA-accelerated basecalling and classification in real time, enabling same-day bloodstream infection pathogen identification. GPU parallelism is the enabling technology for clinical mNGS turnaround within therapeutic decision windows.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

GPU CTC basecalling; GPU k-mer LCA classification; minimap2 GPU alignment to pathogen reference panel; GPU AMR gene DIAMOND search; Bayesian abundance estimation (Bracken); clinical decision threshold scoring; antimicrobial susceptibility genotype prediction.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/real-time-pathogen-identification-clinical.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/real-time-pathogen-identification-clinical.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\real-time-pathogen-identification-clinical.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: NCBI RefSeq pathogen reference sequences (https://ftp.ncbi.nlm.nih.gov/refseq/); CARD AMR database (https://card.mcmaster.ca/); IDseq / Chan Zuckerberg clinical mNGS data (https://czid.org/); NCBI Pathogen Detection (https://www.ncbi.nlm.nih.gov/pathogens/).

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

Dorado (https://github.com/nanoporetech/dorado) — GPU basecaller; MetaCache-GPU (https://arxiv.org/pdf/2106.08150) — GPU real-time classification; DIAMOND (https://github.com/bbuchfink/diamond) — fast AMR gene annotation; CZID/IDseq (https://github.com/chanzuckerberg/czid-workflows) — cloud mNGS pipeline.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

TensorRT for low-latency basecalling; GPU hash tables for k-mer classification; CUDA streams for read-by-read pipeline; cuBLAS for alignment score matrices; real-time CUDA ring buffer for streaming POD5 signal; multi-GPU pipelining of basecall → classify → annotate. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
