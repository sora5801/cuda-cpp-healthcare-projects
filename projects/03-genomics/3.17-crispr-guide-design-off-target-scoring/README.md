# 3.17 — CRISPR Guide Design & Off-Target Scoring

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.17`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

CRISPR–Cas9 cuts DNA wherever the genome matches a 20-letter guide RNA *and* is
followed by an "NGG" PAM — but Cas9 also cuts, more weakly, at near-matching
**off-target** sites, which is the main safety worry when designing a guide. This
project scans a whole genome for a guide, and for every candidate site computes
the number of mismatches and a **CFD (Cutting Frequency Determination)
off-target score** — a product of position-dependent penalty weights that
encodes the crucial *seed effect* (mismatches near the PAM hurt cutting far more
than distal ones). Each genome window is scored independently, so the scan maps
perfectly onto the GPU: **one thread per genome position**, with the guide held
in constant memory. The demo recovers a known on-target site and ranks the
off-targets, then verifies the GPU result against a CPU reference bit-for-bit.

## What this computes & why the GPU helps

Designing effective CRISPR guide RNAs requires genome-wide off-target
assessment: every 20-mer protospacer must be compared against all near-matches in
the genome. For a 3 Gb human genome this is hundreds of millions of candidate
sites *per guide*; tools like Cas-OFFinder use the GPU to evaluate them in
parallel, and the per-site cutting probability is scored with a model like CFD.

**The parallel bottleneck:** sliding a 23-base window across an `L`-base genome
produces `L − 22` candidate windows, and **each window is scored completely
independently** of the others (it reads only its own 23 bases). That is an
embarrassingly parallel *map* over up to 10⁸ windows — exactly the work the GPU
spreads across thousands of threads, one window per thread. The aggregation
(summing and ranking the scores) is comparatively tiny and is done once on the
host.

## The algorithm in brief

- **PAM gate** — a window is a Cas9 target only if its last 3 bases are `NGG`.
- **Mismatch count** — exact integer count of guide-vs-protospacer differences.
- **CFD score** — product of per-position penalty weights over the mismatched
  positions; perfect match = 1.0, each mismatch multiplies in a weight < 1, with
  seed (PAM-proximal) mismatches penalized most.
- **Aggregate** — recover on-target sites (0 mismatches), rank off-targets by
  CFD, and fold the off-target burden into a guide **specificity score**.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/crispr-guide-design-off-target-scoring.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/crispr-guide-design-off-target-scoring.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\crispr-guide-design-off-target-scoring.sln /p:Configuration=Release /p:Platform=x64
```

This project links only the CUDA runtime (`cudart_static.lib`) — no extra CUDA
library is needed, because the per-window scorer is a hand-written kernel (that
is the teaching point).

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/guide_genome_sample.txt`, prints
the recovered on-target and the top-5 off-targets ranked by CFD, shows the
GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/guide_genome_sample.txt` — a **synthetic**
  20-nt guide + a 418-base genome, engineered so the scan recovers exactly one
  on-target and several off-targets whose CFD scores span a wide range. Runs
  offline with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print how to fetch a real
  reference genome (UCSC hg38/mm10) and convert it to the loader format; they
  download nothing and need no credentials.
- **Provenance & license:** see [data/README.md](data/README.md). The CFD weights
  here are a **synthetic teaching model**, not the published Doench-2016 table.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): the
scan recovers the on-target at position 40, lists the off-targets ranked by CFD
(note the 1-mismatch *distal* site at ~0.86 vs the 1-mismatch *seed* site at
0.05 — the seed effect), and ends with `RESULT: PASS`. The program computes
everything on both the **GPU** ([`src/kernels.cu`](src/kernels.cu)) and a **CPU
reference** ([`src/reference_cpu.cpp`](src/reference_cpu.cpp)) and asserts they
agree: integer mismatch counts identical, CFD scores within `1e-12` (in practice
`0.0`). That agreement is the correctness guarantee.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports.
2. [`src/cfd_score.h`](src/cfd_score.h) — the **shared `__host__ __device__`
   scorer**: PAM check + mismatch count + CFD product. The heart of the project;
   compiled by both the CPU and GPU paths so their results match.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
4. [`src/kernels.cu`](src/kernels.cu) — the kernel (one thread per window, guide
   in constant memory) and host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline
   + the data loader.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **Cas-OFFinder** (<https://github.com/snugel/cas-offinder>) — GPU-accelerated
  off-target search with bounded-mismatch **and bulge** enumeration. Study its
  *enumeration* step — the candidate-pruning this teaching version skips.
- **FlashFry** (<https://github.com/aaronmck/FlashFry>) — scalable guide design
  with a precomputed compressed binary index. Study the index that makes
  off-target lookups fast.
- **CRISPOR** (<https://github.com/maximilianh/crisporPaper>,
  <https://crispor.gi.ucsc.edu/>) — an end-to-end on/off-target scoring pipeline;
  the specificity-score formula here follows its MIT-style aggregate.
- **PLM-CRISPR** (<https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12254127/>) —
  protein language model for Cas9-variant activity (the learned-model direction).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**One thread per genome window**, the guide broadcast from **constant memory** —
the "score one query vs N independent items" pattern (PATTERNS.md §1, exemplified
by flagship 1.12 Tanimoto). A grid-stride loop covers an arbitrarily long genome;
the per-window math is shared with the CPU via a `__host__ __device__` core
(PATTERNS.md §2) so verification is exact; the order-dependent reduction is kept
on the host for determinism (PATTERNS.md §3). The catalog also mentions GPU
mismatch *enumeration* (BFS) and cuDNN/transformer efficiency models — those are
described in THEORY §7 as the full-scale extensions this reduced-scope version
omits.

## Exercises

1. **Add the reverse strand.** Real scans check both DNA strands. Add a kernel
   pass over the reverse-complement of the guide (or genome) and merge the hits.
2. **Swap in the real CFD weights.** Replace `cfd_position_weight()` in
   `cfd_score.h` with a position-*and*-mispair table (pass the `(rRNA,dDNA)`
   identity through `score_window`). Confirm CPU and GPU still agree.
3. **Enumerate instead of brute-force.** Pre-filter windows with a bounded
   mismatch budget (skip any window whose PAM fails, then early-exit the mismatch
   loop once it exceeds a threshold) and measure the speed-up on a large genome.
4. **Block-size sweep.** Try 64/128/256/512 threads/block on a big synthetic
   genome (`make_synthetic.py --filler 5000000`) and plot kernel time vs block
   size; explain the occupancy trade-off.
5. **GPU-side top-K.** Replace the host ranking with a GPU partial sort
   (Thrust/CUB) and discuss how to keep it deterministic.

## Limitations & honesty

- **Synthetic everything.** The genome *and* the CFD weights are synthetic
  teaching constructs — the scores have **no biological meaning** and must never
  inform a real CRISPR experiment.
- **Position-only CFD.** Our weight depends only on the mismatch position, not on
  the rRNA:dDNA mispair identity as the real CFD model does (THEORY §7). It
  captures the seed effect but not the full table.
- **Brute force, one strand, no bulges.** We score every forward-strand window
  directly (no enumeration/pruning, no reverse strand, no RNA/DNA bulges, single
  PAM/Cas variant). Production tools do all of these.
- **No on-target efficiency model.** Choosing a *good* guide also needs an
  efficiency prediction (CNN / PLM); that separate GPU workload is out of scope.
- **Timing is a teaching artifact.** On the tiny demo genome the GPU is slower
  than the CPU (launch/copy overhead dominates); the GPU advantage is real only
  at chromosome scale (10⁸ windows).
