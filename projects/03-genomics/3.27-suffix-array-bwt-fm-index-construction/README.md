# 3.27 — Suffix Array / BWT / FM-Index Construction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.27`
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

The BWT (Burrows-Wheeler Transform) and its associated FM-index enable sub-linear text search and are the backbone of short-read aligners (BWA, Bowtie2), assemblers (string graphs), and text compression. Constructing the BWT of a 3 Gb genome involves building the suffix array (SA) then applying the BWT permutation. GPU suffix array construction via parallel prefix-doubling achieves 7.9× speedup over prior GPU skew algorithms, with all n suffixes sorted simultaneously using (log n) radix-sort rounds. At metagenomics or pangenome scale (terabases), GPU construction of a BWT over millions of reads (Big-BWT / ropebwt2) is a research frontier, with CUDA CUDPP's parallel BWT used as a primitives baseline.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Prefix-doubling suffix array construction (DC3/skew algorithm adapted for GPU); radix sort by 2k-character rank pairs; Burrows-Wheeler permutation; FM-index backward step (LF mapping); wavelet tree construction for rank/select; Big-BWT external-memory algorithm.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/suffix-array-bwt-fm-index-construction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/suffix-array-bwt-fm-index-construction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\suffix-array-bwt-fm-index-construction.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: GRCh38 human reference genome — 3 Gb target for BWT construction (https://www.ncbi.nlm.nih.gov/assembly/GCF_000001405.40/); 1000 Genomes read collections for pan-read BWT (https://www.internationalgenome.org/data); NCBI RefSeq complete microbial genomes (https://ftp.ncbi.nlm.nih.gov/refseq/); Human Pangenome sequences for pan-BWT (https://humanpangenome.org/).

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

GPU suffix array prefix-doubling (https://www.researchgate.net/publication/303594470) — fast parallel SA construction on GPU; ropebwt2 (https://github.com/lh3/ropebwt2) — incremental BWT construction (CPU, GPU K40 tested); CUDPP BWT (https://devblogs.nvidia.com/cutting-edge-parallel-algorithms-research-cuda/) — CUDA Data Parallel Primitives BWT; Big-BWT (https://github.com/alshai/Big-BWT) — external-memory BWT for terabase strings.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

thrust::sort_by_key for radix-sort based SA construction; parallel prefix sums (CUB) for rank array update; GPU-resident suffix-rank arrays; custom LF-mapping kernel; persistent warp pattern for backward search. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
