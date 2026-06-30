# 3.26 — GPU BAM Sorting & Deduplication

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.26`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

After short reads are aligned to a reference genome, a sequencing pipeline must
**coordinate-sort** them (put them in genome order) and **mark PCR duplicates**
(collapse the multiple amplified copies of one original DNA fragment to a single
best representative). Both are canonical bottlenecks on terabyte-scale BAM files,
and both are textbook GPU data-parallel primitives: coordinate sort is a **radix
sort** on a packed `(chromosome, position, strand)` key, and duplicate marking is
a **group-by-and-keep-the-best** segmented reduction. This project implements both
on the GPU with **Thrust** (`sort_by_key`, `reduce_by_key`) over a small,
heavily-commented integer read record, verifies the result **exactly** against a
plain-C++ reference, and runs offline on a tiny synthetic sample with a known
duplicate count.

## What this computes & why the GPU helps

Post-alignment BAM sorting (by genomic coordinate) and duplicate read marking are
canonical bottlenecks in sequencing pipelines processing terabyte-scale BAM files.
Coordinate sort is a radix sort on (chromosome, position, strand) keys; GPU radix
sort via Thrust achieves far higher throughput than the `samtools` CPU sort.
Duplicate marking groups reads by `(start, end, orientation)` and keeps only the
highest-base-quality copy — a parallel hash/group-aggregation problem. Parabricks
integrates GPU sort and markdup in its `fq2bam` tool, overlapping the GPU sort with
alignment I/O.

**The parallel bottleneck:** the **sort**. A whole-genome BAM holds ~10⁹ reads; a
CPU comparison sort is $O(n\log n)$ and memory-latency bound. A GPU radix sort is
effectively $O(n)$ (a fixed number of byte-passes) and **bandwidth bound** — the
regime where the GPU's memory bandwidth dominates. Duplicate marking rides on the
same sorted layout: once equal-signature reads are contiguous, finding each group's
best copy is a single segmented reduction.

## The algorithm in brief

- **Radix sort** by a packed `(chromosome, position, strand)` key (`thrust::sort_by_key`).
- **Hash/sort-based read grouping** by the duplicate signature `(ref, pos, strand, mate)`.
- **Picard MarkDuplicates scoring**: keep the read with the largest sum of base
  qualities (ties → lowest id), via a segmented reduction (`thrust::reduce_by_key`).
- (Discussed in THEORY) UMI-aware collapsing and coordinate-index (BAI/CSI)
  construction via parallel prefix.

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

> **Thrust build note.** This project includes Thrust headers (shipped with CUDA),
> so its `.vcxproj` adds two host-compiler flags the integration does not set by
> default: `/Zc:preprocessor` (Thrust/CCCL requires MSVC's conforming
> preprocessor) and `-std=c++17` for nvcc. The Debug config also suppresses nvcc
> diagnostics 20011/20014, which the toolkit's own headers raise under `-G`. All
> three are commented in the `.vcxproj`. No extra `.lib` is linked — Thrust is
> header-only.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (CMake build)
```

The demo builds if needed, runs on `data/sample/reads_sample.txt`, prints the
sorted-order digest and duplicate count, shows the **GPU-vs-CPU exact-match**
check, and prints a timing line on stderr.

## Data

- **Sample (committed):** `data/sample/reads_sample.txt` — 2,000 synthetic aligned
  reads across 4 references with ~358 planted PCR duplicates, so the demo runs with
  zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` (prints where to fetch real
  BAMs and how to convert them; never bypasses credentials).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: 1000 Genomes WGS BAM archives (<https://www.internationalgenome.org/data>);
TCGA cancer WGS BAM files (<https://portal.gdc.cancer.gov/>); ENCODE ChIP-seq BAM
(<https://www.encodeproject.org/>); ICGC PCAWG BAMs (<https://dcc.icgc.org/>).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The
program sorts and dedups on both the **GPU** (`src/kernels.cu`) and a **CPU
reference** (`src/reference_cpu.cpp`) and asserts they agree **exactly** — every
key, position, and flag is an integer with a total order, so verification needs
**no tolerance**. A second, independent check confirms the dedup count equals the
number of duplicates the generator planted (**358**), validating the science, not
just CPU==GPU agreement. `RESULT: PASS` means both checks held.

## Code tour

Read in this order:

1. [`src/bam.h`](src/bam.h) — the shared `__host__ __device__` record + key/compare
   math (the one source of truth both CPU and GPU call).
2. [`src/main.cu`](src/main.cu) — loads reads, runs CPU + GPU, verifies exactly, reports.
3. [`src/reference_cpu.h`](src/reference_cpu.h) / [`src/reference_cpu.cpp`](src/reference_cpu.cpp)
   — the trusted serial baseline (`std::sort` + hash-map grouping).
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the Thrust idea.
5. [`src/kernels.cu`](src/kernels.cu) — the map kernels and the Thrust sort/reduce pipeline.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **Parabricks `fq2bam` / bamsort** (<https://docs.nvidia.com/clara/parabricks/latest/>)
  — integrated GPU BAM sort + dedup; the research-grade tool this models. Learn how
  it overlaps sort with alignment I/O.
- **biobambam2** (<https://github.com/gt1/biobambam2>) — CPU sort/dedup reference
  with parallel threads; study its duplicate-signature definition.
- **Samtools** (<https://github.com/samtools/samtools>) — the canonical CPU BAM
  toolkit (`sort`, `markdup`); the behaviour we mimic.
- **FastDup** (<https://arxiv.org/pdf/2505.06127>) — speculation-and-test GPU
  duplicate marking; a more advanced GPU dedup scheme.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

`thrust::sort_by_key` for the radix coordinate sort; sort-then-`thrust::reduce_by_key`
for per-group best-quality selection (a segmented reduction = "GROUP BY signature,
keep argmax(quality)"); a prefix-scan to assign group ids; and atomic-free map
kernels for key computation and flag writeback. This is the **sort + segmented
reduce** pattern (PATTERNS.md §1 group-aggregate, §5 library-without-a-black-box).
A production tool would add CUDA managed memory for BAM record streaming and a
multi-GPU shard-and-merge — noted in THEORY.

## Exercises

1. **Widen the keys.** Replace the 24/24/1/15-bit packing with a 128-bit key (or a
   Thrust zip-iterator tuple comparator) so real 32-bit positions fit. Verify the
   sort still matches the CPU.
2. **Library-size complexity.** Sweep `--n` from 10³ to 10⁷ in `make_synthetic.py`
   and plot the GPU sort time vs. the CPU sort time. At what read count does the
   GPU overtake the CPU? Tie it back to "launch-bound vs. bandwidth-bound".
3. **Optical vs. PCR duplicates.** Add a synthetic "tile/coordinate" field and a
   distance threshold, and split the duplicate count into optical and PCR — closer
   to what Picard reports.
4. **Index construction.** Build a BAI-style coarse index (counts per fixed genomic
   bin) from the sorted positions using a single `thrust::inclusive_scan`.
5. **UMI-aware collapsing.** Add a synthetic UMI to the signature and collapse only
   reads that share both the fragment ends *and* the UMI.

## Limitations & honesty

- **Reduced-scope teaching model.** We operate on a flat in-memory integer record,
  **not** a real BGZF/BAM file with CIGAR strings, clipping, read pairs, or a
  BAI/CSI index. THEORY.md "Where this sits in the real world" lists every
  simplification.
- **Synthetic data.** The sample is randomly generated reads with a deliberately
  planted duplicate structure — **not** real patient sequencing data, and it
  carries no clinical meaning. It exists to make the result interpretable and the
  verification exact.
- **Timing is a teaching artifact, not a benchmark.** On the tiny sample the GPU is
  launch/copy bound and is *not* faster than the CPU; the radix-sort advantage
  appears only at large read counts. The demo says so on stderr.
- **Not a duplicate-marking authority.** The signature and scoring are simplified
  versions of Picard's; do not use this output for any real analysis.
