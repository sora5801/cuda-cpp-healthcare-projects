# 3.2 — Short-Read Mapping / Alignment

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟢 Beginner · Established** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.2`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

A DNA sequencer does not hand you a genome — it hands you hundreds of millions of
short fragments ("reads", ~50–300 bases each). Before anything biological can be
learned, every read must be **mapped**: placed at the position in a reference
genome it most likely came from. This project builds the heart of a short-read
mapper — the **seed-and-extend** algorithm — and runs it on the GPU with the
simplest, most honest parallel structure: **one read per thread**. We map a batch
of synthetic reads against a small reference, recover each read's true position
and its mismatch count, and verify the GPU result against a plain-C++ reference
**exactly** (integer equality).

## What this computes & why the GPU helps

Short-read mapping first **seeds** candidate positions in a reference genome index
(an FM-index or hash table), then **extends** each seed with a banded Smith-
Waterman alignment. At whole-genome scale (30× coverage ≈ 900 M reads for human),
the seed-extension and CIGAR-string generation steps dominate runtime. GPU
acceleration batches thousands of read-to-reference extensions simultaneously.
NVIDIA Parabricks (v4.7, 2025) completes a 30× WGS in under 10 minutes on an
H100, vs. >30 hours for CPU BWA-MEM, by reimplementing BWA-MEM's seed-chain-extend
pipeline in CUDA.

**The parallel bottleneck:** seed extension — scoring each read against its
candidate reference windows. Reads are **mutually independent** (read 7's best
position does not depend on read 3's), so the work is embarrassingly parallel
across reads. We assign **one GPU thread per read**; all reads are mapped in a
single launch. This is exactly the structure that lets GPUs turn a >30-hour CPU
job into minutes — the dominant cost is per-read extension, and there are
hundreds of millions of independent reads to spread across the device.

## The algorithm in brief

- **Index** the reference: enumerate every length-`k` window (k-mer), pack each
  into a 2-bit-per-base integer code, and **sort** the codes so a seed can be
  found by binary search (a GPU-friendly stand-in for an FM-index/hash table).
- **Seed:** take each read's leading `k`-mer and binary-search the index to get
  every reference offset where that exact k-mer occurs (the candidate positions).
- **Extend:** at each candidate, lay the whole read against the reference and
  score it **ungapped** (+1 per matching base, −1 per mismatch); keep the best
  (highest score; ties broken by lowest offset).
- **Report** each read's mapped position, score, and mismatch count (a CIGAR-like
  `<L>M` summary).

Key algorithms from the catalog: FM-index / BWT backward search; seed chaining
(sparse DP); banded Smith-Waterman extension; CIGAR encoding; minimizer seeding.
This teaching version implements the **sorted-k-mer-index + ungapped-extension**
slice; see [THEORY.md](THEORY.md) for what the full pipeline adds and how the
gapped-SW wavefront ([project 3.01](../3.01-smith-waterman-needleman-wunsch-alignment/))
slots into the "extend" step.

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

No extra CUDA libraries are linked — only the CUDA runtime and our own kernels.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/reads_sample.txt`, prints the
per-read mapping table, shows the GPU-vs-CPU agreement check, and prints a timing
line to stderr.

## Data

- **Sample (committed):** `data/sample/reads_sample.txt` — a tiny **synthetic**
  reference (240 bp) + 11 reads (40 bp), so the demo runs with zero downloads.
  Ten reads are sampled from known, evenly spaced positions with 0–3 point
  mutations (the embedded known answer); one is random noise that maps nowhere.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print instructions and
  links for the real datasets (they never bypass registration).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: 1000 Genomes Project (https://www.internationalgenome.org/data);
Genome in a Bottle (GiaB) NA12878 / HG002 (https://www.nist.gov/programs-projects/genome-bottle);
SRA FASTQ archives (https://www.ncbi.nlm.nih.gov/sra);
ENCODE ChIP/RNA-seq FASTQs (https://www.encodeproject.org/).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): each
read maps to its true reference position with score `40 − 2·(mismatches)`, the
noise read is `UNMAPPED`, and the final line reads
`RESULT: PASS (GPU matches CPU exactly on every read)`. The program maps every
read on both the **GPU** (`src/kernels.cu`) and a **CPU reference**
(`src/reference_cpu.cpp`) and asserts they agree on every read's
`(position, score, mismatches)` by **exact integer equality** — that agreement is
the correctness guarantee (no floating point is involved, so the tolerance is
zero; see THEORY §6).

## Code tour

Read in this order:

1. [`src/reference_cpu.h`](src/reference_cpu.h) — the data model and the shared
   `__host__ __device__` scoring core (`kmer_code`, `kmer_equal_range`,
   `score_window`) that both CPU and GPU run.
2. [`src/main.cu`](src/main.cu) — loads data, builds the index, runs CPU + GPU,
   verifies, reports.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-read idea.
4. [`src/kernels.cu`](src/kernels.cu) — the kernel and host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the loader, index builder,
   and trusted serial mapper.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **NVIDIA Parabricks** (<https://docs.nvidia.com/clara/parabricks/latest/>) —
  GPU BWA-MEM + GATK; study the production seed-chain-extend pipeline this project
  miniaturizes.
- **CUSHAW2-GPU / CUSHAW3** (<https://github.com/asbschmidt/CUSHAW3>) — banded SW
  seed extension on the GPU; the natural "next step" beyond our ungapped extend.
- **Scrooge** (<https://github.com/CMU-SAFARI/Scrooge>) — GPU/CPU co-designed
  aligner; learn how work is split between host and device.
- **GenomeWorks** (<https://github.com/NVIDIA-Genomics-Research/GenomeWorks>) —
  pairwise/overlap kernels underpinning mapping and assembly.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Independent jobs, one thread per read** (the `1.12` Tanimoto pattern from
[docs/PATTERNS.md](../../../docs/PATTERNS.md) §1), combined with the shared
`__host__ __device__` core (§2) for exact CPU/GPU parity. Each thread runs a
grid-stride loop over reads; for its read it computes the seed code, binary-
searches the sorted reference index, scores every candidate window in registers,
and writes the best `(pos, score, mism)` to its own output slot. **No shared
memory, no atomics** — the per-read max-reduction is private to the thread and the
outputs are disjoint, so the kernel is automatically deterministic. The catalog
also lists Thrust (seed sorting), cuSPARSE (index look-ups), and NCCL (multi-GPU);
we sort the small index on the host and explain in THEORY §7 where those scale.

## Exercises

1. **Gapped extension.** Replace the ungapped `score_window` with a banded
   Smith-Waterman (reuse the wavefront idea from [project 3.01](../3.01-smith-waterman-needleman-wunsch-alignment/))
   so reads with indels still map. How does the CIGAR string change?
2. **Reverse-complement strand.** Real reads can come from either DNA strand. Add
   a second seed lookup using the read's reverse complement and report the strand.
3. **Multi-seed / chaining.** Use several seeds per read (not just the leading
   k-mer) and require them to be co-linear before extending — this is how real
   mappers reject spurious single-seed hits.
4. **Minimizer seeding.** Index only the minimizers of each window instead of all
   k-mers; measure how much smaller the index becomes and what mapping you lose.
5. **Warp-per-read extension.** For long reads, give each read a whole warp and
   parallelize `score_window` across lanes with a warp reduction. Compare timing.

## Limitations & honesty

This is a deliberately **reduced-scope teaching version**:

- **Ungapped extension only.** We score substitutions, not insertions/deletions,
  so reads with indels would mis-score. Real mappers use banded gapped SW.
- **Sorted-array index, not an FM-index.** We sort the reference k-mers and binary-
  search them. A true FM-index (BWT + rank/select) is far more memory-efficient
  for a whole human genome and supports variable-length seeds — see THEORY §7.
- **Exact, fixed-length leading seed; uniform read length.** One seed per read,
  taken from its first `k` bases; all reads must be the same length. Production
  tools handle variable lengths, multiple seeds, and seed errors.
- **Synthetic data.** The sample is generated with a known answer and carries **no
  biological meaning**. Nothing here is validated for clinical or diagnostic use.
- **Timing is a teaching artifact.** On this tiny batch the GPU is launch/copy
  bound and can be slower than the CPU; the GPU's edge appears at millions of
  reads (CLAUDE.md §12).
