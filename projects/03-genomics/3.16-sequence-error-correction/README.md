# 3.16 — Sequence Error Correction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.16`
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

Error correction removes sequencing artefacts before assembly. For short reads, the dominant method is k-mer spectrum analysis: k-mers below a coverage threshold are likely errors; correcting a base changes the read k-mer into a trusted one. For long reads (ONT, PacBio CLR), self-correction aligns multiple raw reads against each other and computes a consensus. CARE (https://github.com/fkallen/CARE) is a CUDA-accelerated short-read error corrector that keeps the k-mer hash table in GPU memory and processes millions of reads per second. GPU-accelerated partial-order alignment (POA) for long-read correction is implemented in GenomeWorks racon-GPU.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

K-mer spectrum analysis (trusted-k-mer correction); Bloom filter for inexact k-mer membership; multiple sequence alignment (POA / MSA) for long-read consensus; BFC (BWT-based correction); de Bruijn graph compaction for error pruning; expectation-maximisation for error model learning.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/sequence-error-correction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/sequence-error-correction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\sequence-error-correction.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: GAGE short-read datasets — benchmark reads with known errors (http://gage.cbcb.umd.edu/); GiaB HG001-HG007 — truth-set comparison for corrected reads (https://www.nist.gov/programs-projects/genome-bottle); ONT long-read SRA archives (https://www.ncbi.nlm.nih.gov/sra); PacBio CLR SRA datasets — high-error long reads (https://www.ncbi.nlm.nih.gov/sra).

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

CARE (https://github.com/fkallen/CARE) — CUDA short-read error corrector, GPU hash tables, Pascal+ required; racon-GPU (https://github.com/NVIDIA-Genomics-Research/racon-gpu) — GPU POA polishing/correction; CONSENT (https://github.com/morispi/CONSENT) — long-read self-correction via local De Bruijn graphs (CPU, GPU POA target); Medaka (https://github.com/nanoporetech/medaka) — RNN-based long-read correction with GPU inference.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

GPU hash tables with atomic CAS for k-mer counting; warp-level vote for consensus base determination; cuBLAS / custom GEMM for MSA scoring; one CUDA block per read during POA alignment; batched kernel launches across millions of reads. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
