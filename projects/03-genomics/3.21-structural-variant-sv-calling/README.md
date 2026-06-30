# 3.21 — Structural Variant (SV) Calling

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.21`
>
> _Educational only — not for clinical use (see CLAUDE.md §8). This is a **reduced-scope teaching version** (CLAUDE.md §13): it calls **deletions** from split reads; the full multi-type, assembly-and-CNN pipeline is described in [THEORY.md](THEORY.md)._

## Summary

A **structural variant (SV)** is a large genomic change — a deletion, insertion,
inversion, or translocation of ≥50 bp. Long reads that span an SV breakpoint
align to the reference in two pieces ("split reads"); the jump between the pieces
reveals the variant. This project takes many candidate split reads, **re-aligns
each one precisely** to pinpoint its breakpoint (a tiny banded Smith-Waterman
alignment), then **clusters** the per-read breakpoint estimates that agree into a
single SV call with a supporting-read count and a genotype. Both steps are
embarrassingly parallel across reads, so the GPU does one read per thread; the
clustering is an integer atomic histogram, which makes the GPU result **bit-for-bit
identical** to a plain CPU reference. We run it on a tiny synthetic dataset with a
known planted deletion so you can see the caller recover the truth.

## What this computes & why the GPU helps

Structural variants (deletions, insertions, inversions, translocations ≥50 bp) are detected by read-support signatures: split reads, discordant pairs, and assembly-based breakpoint realignment. GPU acceleration applies at two points: (1) rapid re-alignment of split-read candidates using banded SW to pinpoint breakpoints precisely, and (2) batched deep learning inference (convolutional models on pileup images) to genotype and filter SVs. Sniffles2 uses a fast clustering algorithm for ONT/HiFi; pbsv uses local realignment. GPU-accelerated genotyping (similar to DeepVariant's image-based approach) is an emerging direction for SV filtering at population scale.

**The parallel bottleneck:** a population-scale callset begins with *millions* of
candidate reads, and each read's banded-SW re-alignment + breakpoint refinement is
**independent of every other read**. That is the cost center, and it maps perfectly
to "one read per GPU thread" (PATTERNS.md §1, *independent jobs*). The subsequent
clustering of refined breakpoints into calls is a **scatter-reduction** (many reads
vote into the same breakpoint bin) handled with integer `atomicAdd` (PATTERNS.md §1,
*parallel assign + atomic reduce*; exemplar 11.09).

## The algorithm in brief

- **Per-read breakpoint refinement** — banded Smith-Waterman (`src/sv.h`): slide a
  ±window around the read's raw breakpoint guess, score the read's left flank
  against the reference at each candidate, keep the best position.
- **Breakpoint clustering** — vote each refined breakpoint into a 1-bp histogram
  (integer `atomicAdd` on the GPU; a plain loop on the CPU), and a parallel sum of
  the deletion-length estimates.
- **Peak merging → calls** — greedily emit a call at each local-maximum bin that
  clears the support floor, merging votes within ±`SV_MERGE` bp.
- **Genotype** — an integer variant-allele-fraction rule (support vs. total reads)
  → `0/0`, `0/1`, or `1/1` (no floating point in the decision).

Catalog key algorithms (full scope): split-read alignment and breakpoint
clustering; discordant pair signature scoring; local assembly with miniasm/hifiasm
at breakpoints; convolutional image-based genotyping (DeepSV style); SV merging
across samples (SURVIVOR); genotype likelihood calculation.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/structural-variant-sv-calling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/structural-variant-sv-calling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\structural-variant-sv-calling.sln /p:Configuration=Release /p:Platform=x64
```

This project links only `cudart_static.lib` (the CUDA runtime) — the banded SW and
the histogram are hand-rolled kernels, which is the more didactic choice here. No
extra CUDA library is needed.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/sv_sample.txt`, prints the SV
call(s), shows the GPU-vs-CPU agreement check, reports whether the planted SV was
recovered, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/sv_sample.txt` — a tiny, offline, **synthetic**
  reference window + 24 candidate reads with a planted deletion, so the demo runs
  with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print pointers to the real
  benchmarks (no credentials are ever bypassed).
- **Provenance & license:** see [data/README.md](data/README.md). The sample is
  labeled synthetic everywhere it appears.

Catalog dataset notes: GiaB SV benchmark (HG002) — gold-standard deletion/insertion/inversion calls (https://www.nist.gov/programs-projects/genome-bottle); PacBio SV benchmark (https://github.com/PacificBiosciences/sv-benchmark); 1000 Genomes SV catalog (https://www.internationalgenome.org/data); ENCODE long-read SV studies (https://www.encodeproject.org/).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt):

```
3.21 -- Structural Variant (SV) Calling
reduced-scope teaching version: deletion calling by split-read
realignment (banded SW) + breakpoint clustering on SYNTHETIC data
reference length = 240 bp, candidate reads = 24, min support = 3
SV calls (sorted by breakpoint): 1
  DEL  bp=120  len=50  support=18  GT=1/1
planted truth: bp=120 len=50  -> recovered: YES
RESULT: PASS (GPU histogram+calls match CPU exactly)
```

The program runs the pipeline on both the **GPU** (`src/kernels.cu`) and a **CPU
reference** (`src/reference_cpu.cpp`) and asserts the breakpoint histograms and the
emitted call lists are **exactly equal**. Because both sides run the same integer
math from `src/sv.h` and accumulate with commuting integer (atomic) adds, the
tolerance is **zero** — any mismatch is a real bug, not floating-point noise.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports.
2. [`src/sv.h`](src/sv.h) — the shared `__host__ __device__` math: banded SW,
   breakpoint refinement, binning, integer genotype. **This is the heart.**
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
4. [`src/kernels.cu`](src/kernels.cu) — the refine+vote kernel and host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline +
   the shared histogram→calls merge.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

Sniffles2 (https://github.com/fritzsedlazeck/Sniffles) — fast ONT/HiFi SV caller; PBSV (https://github.com/PacificBiosciences/pbsv) — PacBio SV caller; cuteSV (https://github.com/tjiangHIT/cuteSV) — clustering-based SV caller; NGSEP (https://github.com/NGSEP/NGSEPcore) — variant calling suite with GPU-amenable CNN scoring.

- **Sniffles2 / cuteSV** show production-grade *breakpoint clustering* across many
  reads — the clustering step here is a stripped-down version of their idea.
- **pbsv** shows *local realignment* at breakpoints — what our banded SW models.
- **DeepVariant / DeepSV** show CNN pileup-image genotyping — the "full version"
  filtering step described in THEORY but not implemented here.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Independent jobs (one read per thread) + atomic histogram reduction.** Each GPU
thread re-aligns one read's flank with banded Smith-Waterman (`src/sv.h`) and votes
its refined breakpoint into a shared integer histogram via `atomicAdd`. Catalog
pattern (full scope): *Banded SW CUDA kernels for breakpoint realignment; cuDNN CNN
for SV image genotyping; batched pileup image inference; thrust for read cluster
sorting; multi-GPU for population-scale SV genotyping.*

## Exercises

1. **Scale it.** Run `python scripts/make_synthetic.py --reads 200000` and re-run;
   watch the GPU's relative advantage grow as the per-read realignment count rises
   (the timing is launch-bound at 24 reads).
2. **Add a second SV.** Extend `make_synthetic.py` to plant two deletions at
   different breakpoints and confirm the clustering emits two distinct calls.
3. **Heterozygous genotypes.** Add reads that span the locus but support the
   *reference* allele (no deletion), so `total > support`, and watch the integer
   VAF rule in `sv_geno_from_vaf` flip the call from `1/1` to `0/1`.
4. **Shared-memory reference.** The reference window is read by every thread; cache
   it (or a tile of it) in `__shared__` memory and measure the effect on a larger run.
5. **Affine gaps.** Replace the linear gap penalty in `sv_banded_sw` with an
   affine (open + extend) model and discuss how that changes breakpoint precision.

## Limitations & honesty

- **Reduced scope.** This calls **deletions only**, from pre-extracted split-read
  flanks. It does **not** parse BAM/CRAM, do discordant-pair signatures, local
  assembly, insertions/inversions/translocations, or CNN genotyping — those are
  described in [THEORY.md](THEORY.md) under "Where this sits in the real world".
- **Synthetic data.** The committed sample is generated with a fixed seed and is
  labeled synthetic everywhere. No real genome is included, and no output here is
  clinically valid.
- **Toy genotype.** The genotype uses fixed integer VAF cutoffs, not a proper
  genotype-likelihood model; in the synthetic mix `total` is dominated by
  supporting reads, so the planted "het" reads as `1/1`. This is a teaching
  simplification, called out in the code and THEORY §6.
- **Timing is a teaching artifact, not a benchmark** (CLAUDE.md §12): at 24 reads
  the GPU is launch/copy-bound and may be *slower* than the CPU; the win appears at
  scale.
