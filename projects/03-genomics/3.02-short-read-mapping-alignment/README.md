# 3.2 — Short-Read Mapping / Alignment

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟢 Beginner · Established** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.2`
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

Short-read mapping (50–300 bp Illumina reads) first seeds candidate positions in a reference genome index (FM-index or hash table), then extends seeds with banded SW. At whole-genome scale (30× coverage ≈ 900 M reads for human), the seed-extension and CIGAR-string generation steps dominate runtime. GPU acceleration batches thousands of read-to-reference extensions simultaneously, each assigned a CUDA thread block with shared-memory score matrix, while FM-index backward search runs as a parallel BFS across thread groups. NVIDIA Parabricks (v4.7, 2025) completes a 30× WGS in under 10 minutes on an H100, vs. >30 hours CPU BWA-MEM, by reimplementing BWA-MEM's seed-chain-extend pipeline in CUDA.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

FM-index / BWT backward search; seed chaining (sparse DP); banded Smith-Waterman extension; CIGAR encoding; markduplicates hashing; Burrows-Wheeler transform; seeding by minimisers.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/short-read-mapping-alignment.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/short-read-mapping-alignment.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\short-read-mapping-alignment.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: 1000 Genomes Project — 2504 human WGS samples, short reads (https://www.internationalgenome.org/data); Genome in a Bottle (GiaB) NA12878 / HG002 — benchmark short-read WGS datasets (https://www.nist.gov/programs-projects/genome-bottle); SRA FASTQ archives — petabyte-scale short reads (https://www.ncbi.nlm.nih.gov/sra); ENCODE ChIP/RNA-seq FASTQs — curated short-read functional data (https://www.encodeproject.org/).

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

NVIDIA Parabricks (https://docs.nvidia.com/clara/parabricks/latest/) — GPU-accelerated BWA-MEM + GATK pipeline, 50× faster than CPU; CUSHAW2-GPU (https://github.com/asbschmidt/CUSHAW3) — banded SW seed extension on GPU; Scrooge (https://github.com/CMU-SAFARI/Scrooge) — GPU/CPU co-designed aligner; GenomeWorks (https://github.com/NVIDIA-Genomics-Research/GenomeWorks) — pairwise overlap kernels underpinning mapping.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuSPARSE (index look-ups); thrust (sorting seeds); custom banded-SW kernels with shared-memory tiling; persistent warp-per-read extension; multi-GPU data parallelism via NCCL. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
