# 3.7 — BLAST-Style Homology Search

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟢 Beginner · Established** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.7`
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

Homology search finds sequences in a database that are evolutionarily related to a query, using seed-filter-extend logic (BLAST) or k-mer prefiltering + ungapped alignment (MMseqs2 / DIAMOND). At the scale of AlphaFold2 structure prediction (MSA search dominates 70–90% of total inference time), GPU acceleration is transformative. MMseqs2-GPU (2025, Nature Methods) replaces the CPU k-mer prefilter with a GPU-parallel gapless scoring pass across all database sequences simultaneously, achieving 20× speedup and 71× cost reduction vs. 128-core CPU. The bottleneck parallelised is the embarrassingly parallel pairwise k-mer match scanning across millions of database sequences per query batch.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

K-mer prefilter seeding; gapless diagonal scoring; Smith-Waterman extension (affine gaps); profile-profile scoring (PSI-BLAST); iterative profile construction; DIAMOND's double-indexed seed matching.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/blast-style-homology-search.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/blast-style-homology-search.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\blast-style-homology-search.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: UniRef50/90 — clustered UniProt sequences for homology (https://www.uniprot.org/help/uniref); NCBI nr protein database (https://ftp.ncbi.nlm.nih.gov/blast/db/); PDB70 — representative PDB sequences (https://www.rcsb.org/downloads); Pfam — protein family HMM database (https://www.ebi.ac.uk/interpro/download/).

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

MMseqs2 + GPU branch (https://github.com/soedinglab/MMseqs2) — official repo with GPU support in 2025 release; DIAMOND (https://github.com/bbuchfink/diamond) — ultra-fast protein aligner (CPU baseline); CUDASW4 (https://github.com/asbschmidt/CUDASW4) — full SW on GPU for deep alignments; NVIDIA NIM MMseqs2 microservice (https://developer.nvidia.com/blog/accelerated-sequence-alignment-for-protein-design-with-mmseqs2-and-nvidia-nim/) — cloud-API GPU search.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom CUDA gapless scoring kernels (one warp per query-target pair); batched SW extension with shared memory; GPU hash table for seed look-ups; multi-GPU data parallelism across database shards; CUDA streams for overlapping I/O and compute. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
