# 3.26 — GPU BAM Sorting & Deduplication

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.26`
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

Post-alignment BAM sorting (by genomic coordinate) and duplicate read marking are canonical bottlenecks in sequencing pipelines processing terabyte-scale BAM files. Coordinate sort is a radix sort on (chromosome, position, strand) keys; GPU radix sort via thrust achieves far higher throughput than samtools CPU sort. Duplicate marking requires grouping reads by (start, end, orientation) and keeping only the highest-base-quality copy; this is a parallel hash-aggregation problem ideal for GPU hash maps. Parabricks integrates GPU sort and markdup in its fq2bam tool, running in the same 6-minute wall time as the alignment step by overlapping GPU sort with alignment I/O.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Radix sort by (chromosome, position) key; hash-based read grouping for duplicate detection; Picard MarkDuplicates scoring (sum base quality); UMI-aware duplicate collapsing; coordinate index (BAI/CSI) construction via parallel prefix.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/gpu-bam-sorting-deduplication.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/gpu-bam-sorting-deduplication.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\gpu-bam-sorting-deduplication.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: 1000 Genomes WGS BAM archives (https://www.internationalgenome.org/data); TCGA cancer WGS BAM files (https://portal.gdc.cancer.gov/); ENCODE ChIP-seq BAM (https://www.encodeproject.org/); ICGC PCAWG BAMs (https://dcc.icgc.org/).

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

Parabricks fq2bam / bamsort (https://docs.nvidia.com/clara/parabricks/latest/) — integrated GPU BAM sort + dedup; biobambam2 (https://github.com/gt1/biobambam2) — CPU sort/dedup reference with parallel threads; Samtools (https://github.com/samtools/samtools) — CPU BAM toolkit; FastDup (https://arxiv.org/pdf/2505.06127) — speculation-and-test GPU duplicate marking.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

thrust::sort_by_key for radix coordinate sort; GPU robin-hood hash map for duplicate grouping; thrust::reduce_by_key for per-group best-quality selection; CUDA managed memory for BAM record streaming; multi-GPU shard-and-merge pattern. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
