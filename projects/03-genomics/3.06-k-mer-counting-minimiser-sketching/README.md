# 3.6 — k-mer Counting & Minimiser Sketching

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟢 Beginner · Established** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.6`
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

k-mer counting determines the frequency of every length-k substring in a read set, foundational to genome-size estimation, error detection, assembly, and metagenomics. For a 30× human genome (~270 Gb of sequence, k=21), the table has ~4 billion distinct k-mers; efficient parallel hashing and atomic counting saturate GPU memory bandwidth. Gerbil uses GPU-resident hash tables and achieves >10× speed over Jellyfish. Minimiser sketching (selecting a canonical subset of k-mers per window) reduces data by ~5× and enables the MinHash / HyperMinHash distance computations used in species typing; all operations parallelise across reads with one GPU thread per minimiser.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Radix-sort-based k-mer canonicalisation; GPU hash table with cuckoo / Robin Hood probing; count-min sketch for approximate counting; minimiser extraction (window function); MinHash / Jaccard distance estimation; HyperLogLog cardinality estimation.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/k-mer-counting-minimiser-sketching.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/k-mer-counting-minimiser-sketching.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\k-mer-counting-minimiser-sketching.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Illumina WGS of NA12878 — human reference dataset (https://www.ncbi.nlm.nih.gov/sra/SRR622457); GAGE benchmark — multi-species short reads for assembly tools (http://gage.cbcb.umd.edu/); GenomeTrakr pathogen WGS — bacterial surveillance reads (https://www.ncbi.nlm.nih.gov/bioproject/PRJNA183844); Sequence Read Archive (SRA) — global repository (https://www.ncbi.nlm.nih.gov/sra).

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

Gerbil (https://github.com/uni-halle/gerbil) — k-mer counter with GPU support; KMC3 (https://github.com/refresh-bio/KMC) — disk-I/O efficient CPU k-mer counter (GPU comparison baseline); Jellyfish (https://github.com/gmarcais/Jellyfish) — lock-free hash k-mer counter; GenomeScope2 (https://github.com/tbenavi1/genomescope2.0) — genome profiling from k-mer histograms.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CUDA atomic operations (atomicAdd for count tables); thrust::sort_by_key for radix sort; warp-level ballot and shuffle for minimiser window reduction; cuRAND for sketch initialisation. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
