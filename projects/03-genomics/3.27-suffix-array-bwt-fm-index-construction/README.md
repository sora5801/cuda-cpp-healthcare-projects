# 3.27 — Suffix Array / BWT / FM-Index Construction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.27`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

This project builds the three data structures at the heart of every modern
short-read aligner (BWA, Bowtie2) and many compressors: the **suffix array (SA)**,
the **Burrows-Wheeler transform (BWT)**, and a tiny **FM-index** that answers
"how many times does this pattern occur?" in time proportional to the *pattern*
length, not the genome length. We construct the suffix array on the GPU by
**prefix doubling** — repeatedly sorting all suffixes by a packed 64-bit
rank-pair key using a hand-rolled, deterministic **radix sort** — then derive the
BWT and run an FM-index **backward search**. A plain serial CPU version computes
the same structures, and the demo asserts the GPU and CPU suffix arrays are
**bit-for-bit identical** (they are integer permutations, so the tolerance is
exactly zero). The committed input is a tiny, clearly-synthetic DNA string.

## What this computes & why the GPU helps

The BWT (Burrows-Wheeler Transform) and its associated FM-index enable
sub-linear text search and are the backbone of short-read aligners (BWA,
Bowtie2), assemblers (string graphs), and text compression. Constructing the
BWT of a genome means first building the **suffix array** — the sorted order of
all `n` suffixes of the text — then applying the BWT permutation. The suffix
array is the expensive part: a naive comparison sort is `O(n^2 log n)` because
each suffix comparison is `O(n)`. **Prefix doubling** turns it into
`O(n log^2 n)` (or `O(n log n)` with radix sort) by sorting suffixes by their
first `2^r` characters in round `r`, reusing the previous round's ranks so each
comparison is `O(1)`.

**The parallel bottleneck:** in each doubling round, *all `n` suffixes are sorted
at once* by an independent `(rank_i, rank_{i+k})` key pair. Sorting is the
dominant cost, and it is embarrassingly parallel as a **radix sort over
(key, suffix-index) records**. GPU prefix-doubling sorts all suffixes
simultaneously in `O(log n)` radix-sort rounds; published work reports ~7.9×
over earlier GPU skew (DC3) implementations. That sort — not the per-element
arithmetic — is exactly what the GPU accelerates here.

## The algorithm in brief

- **Prefix-doubling suffix array:** rank each suffix by its first character, then
  in rounds `k = 1, 2, 4, …` sort by the rank pair `(rank[i], rank[i+k])` and
  renumber, doubling the known-ordered prefix length each round.
- **Radix sort by rank pairs:** pack each pair into one 64-bit key and LSD
  radix-sort the `(key, suffix-index)` records (8 passes of 8-bit digits).
- **Burrows-Wheeler permutation:** `BWT[i] = text[(SA[i]-1+n) mod n]`.
- **FM-index backward search (LF mapping):** the `C[]` table + occurrence counts
  let us count pattern occurrences right-to-left in `O(|pattern|)`.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation, including how `thrust::sort_by_key` / CUB would replace the
hand-rolled radix sort in production.

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

This project links only `cudart_static.lib` (the CUDA runtime). The radix sort,
prefix scan, and rank renumbering are all hand-written kernels — no extra CUDA
library is needed, which keeps the build identical to every other project in the
collection.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/dna_sample.txt`, prints the
suffix array / BWT / FM-search result, shows the GPU-vs-CPU agreement check, and
prints a timing line (on stderr).

## Data

- **Sample (committed):** `data/sample/dna_sample.txt` — a tiny (60-base),
  clearly-**synthetic** DNA string with a planted repeated motif, so the demo
  runs with zero downloads and produces an interpretable answer.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` (documented, idempotent;
  they print instructions and links and never bypass any credential gate).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: GRCh38 human reference genome — 3 Gb target for BWT
construction (https://www.ncbi.nlm.nih.gov/assembly/GCF_000001405.40/);
1000 Genomes read collections for pan-read BWT
(https://www.internationalgenome.org/data); NCBI RefSeq complete microbial
genomes (https://ftp.ncbi.nlm.nih.gov/refseq/); Human Pangenome sequences for
pan-BWT (https://humanpangenome.org/).

## Expected output

Success looks like `demo/expected_output.txt`:

```
3.27 -- Suffix Array / BWT / FM-Index Construction
text length (with $ sentinel): 61
suffix array SA[0:12] = 60 9 0 20 40 10 30 50 38 36 18 26
BWT[0:32] = GT$TGACGGCGTTCTTGCCGTAAAAAATCAGA
FM-index backward search: pattern "ACG" occurs 6 time(s)
verify: SA mismatches=0  BWT match=yes  FM match=yes
RESULT: PASS (GPU suffix array matches CPU exactly, tol=0)
```

The program computes the SA/BWT/FM result on both the **GPU** (`src/kernels.cu`)
and a **CPU reference** (`src/reference_cpu.cpp`) and asserts they agree
**exactly** (the suffix array is an integer permutation — there is no floating
point, so any difference at all is a bug). `SA[0]=60` is the `$` sentinel suffix
(always sorts first); the `BWT` is the block-sorted last column; and the planted
motif makes `"ACG"` occur 6 times, recovered by FM-index backward search.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the DNA text, runs CPU + GPU, verifies
   (SA mismatches, BWT equality, FM count), reports.
2. [`src/sa_core.h`](src/sa_core.h) — the **shared** `__host__ __device__` key
   math (`pack_key`, `char_to_code`) both the CPU and GPU call, so their sort
   keys are identical.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the radix-sort
   doubling idea.
4. [`src/kernels.cu`](src/kernels.cu) — the kernels (build-keys, histogram,
   scatter, flag, inclusive scan, write-ranks) and the host orchestration loop.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline
   (`std::stable_sort` doubling) plus the shared BWT/FM helpers.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **GPU suffix array prefix-doubling**
  (https://www.researchgate.net/publication/303594470) — the fast parallel SA
  construction this project simplifies; study how it makes the radix sort and
  rank update fully GPU-resident.
- **ropebwt2** (https://github.com/lh3/ropebwt2) — incremental BWT construction
  over read collections (CPU, GPU K40 tested); learn the rope/B+-tree trick for
  growing a BWT without rebuilding.
- **CUDPP BWT**
  (https://devblogs.nvidia.com/cutting-edge-parallel-algorithms-research-cuda/) —
  CUDA Data Parallel Primitives' BWT; the canonical "use library primitives"
  baseline (its sort/scan are what our hand-rolled kernels stand in for).
- **Big-BWT** (https://github.com/alshai/Big-BWT) — external-memory BWT for
  terabase strings via prefix-free parsing; learn how construction scales past RAM.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Radix-sort based SA construction. The catalog calls for `thrust::sort_by_key`
plus CUB prefix sums; to keep this a **no-black-box** teaching project, we
hand-roll exactly those primitives: an LSD radix sort (histogram → exclusive
scan → stable scatter) over `(key, suffix-index)` records, an inclusive prefix
scan for rank renumbering, and a custom kernel chain for the doubling loop.
THEORY.md explains where `thrust`/CUB would slot in and why they win at scale.

## Exercises

1. **Swap in Thrust.** Replace the hand-rolled radix sort with
   `thrust::stable_sort_by_key` (you will need `-Xcompiler /Zc:preprocessor` on
   MSVC for CUDA 13's CCCL). Confirm the SA is unchanged and compare timings.
2. **Parallelize the scatter.** The radix `scatter_kernel` is single-threaded for
   guaranteed stability. Implement a per-block stable scatter (each block scans
   its own digit histogram, then a global offset) and verify the SA still matches.
3. **Bigger input.** Generate a longer string with
   `python scripts/make_synthetic.py --n 4000` and watch the GPU kernel time grow
   more slowly than the CPU's `O(n log^2 n)` baseline.
4. **Locate, don't just count.** Extend the FM-index to *report positions* (not
   just the count) by sampling the suffix array and walking LF until a sampled row.
5. **Rank dictionary.** Replace the `O(n)`-per-step `Occ()` scan with a
   precomputed occurrence table (a baby wavelet tree), turning backward search
   into `O(|pattern|)` independent of `n`.

## Limitations & honesty

- **Reduced scope, on purpose.** This is a *teaching* suffix-array builder. It
  runs in-core on small strings; production tools (BWA's `bwtsw`, Big-BWT,
  ropebwt2) use external memory, prefix-free parsing, and 2-bit packing to handle
  3 Gb genomes — described in THEORY but not implemented here.
- **The radix scatter is single-threaded** for didactic stability and
  determinism; it is *not* the high-throughput parallel scatter a real GPU sort
  uses. At the demo's `n = 61` the GPU is launch-bound and slower than the CPU —
  the timing is a teaching artifact, never a benchmark (CLAUDE.md §12).
- **Synthetic data only.** `data/sample/dna_sample.txt` is generated, labeled
  synthetic everywhere, and is not a real genome or patient sequence.
- **FM-index `Occ()` is `O(n)` per step** (a plain scan), so backward search here
  is `O(n·|pattern|)`, not the textbook `O(|pattern|)`; the rank-dictionary that
  fixes this is left as Exercise 5.
- **No clinical use.** Nothing here may inform diagnosis or treatment.
