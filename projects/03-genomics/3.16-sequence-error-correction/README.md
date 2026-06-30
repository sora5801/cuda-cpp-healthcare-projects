# 3.16 — Sequence Error Correction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.16`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

DNA sequencers read a genome by sampling many short, overlapping fragments
("reads"), and every base call carries a small error probability — so raw reads
contain wrong bases. This project implements **k-mer spectrum error correction**,
the dominant short-read method: count every length-*k* substring across all reads,
declare the frequent ones "trusted" (real genome) and the rare ones suspect, then
fix bases that a single substitution turns from untrusted into trusted. It is a
two-phase GPU pipeline — a **parallel histogram** (atomics) to build the spectrum,
then **one independent thread per read** to correct — and it ships with synthetic
reads whose ground truth lets you watch errors fall from 132 to 39 (≈70% removed).

## What this computes & why the GPU helps

Error correction removes sequencing artefacts before assembly. For short reads,
the dominant method is k-mer spectrum analysis: k-mers below a coverage threshold
are likely errors; correcting a base changes the read k-mer into a trusted one.
For long reads (ONT, PacBio CLR), self-correction aligns multiple raw reads
against each other and computes a consensus. CARE keeps the k-mer hash table in
GPU memory and processes millions of reads per second; racon-GPU does
partial-order alignment for long-read correction.

**The parallel bottleneck:** both phases scale with the **number of reads**, which
is enormous (10⁸–10⁹ reads per run). Phase 1 (counting k-mers) is a giant
histogram — billions of independent `atomicAdd`s into a shared table. Phase 2
(correcting reads) is *embarrassingly parallel*: each read is corrected
independently against the now-frozen spectrum. Both map directly onto "one thread
per read", which is exactly what the GPU is built for.

## The algorithm in brief

- **K-mer spectrum analysis (trusted-k-mer correction)** — count k-mers; trust
  those with count ≥ T; substitute suspect bases to recover trusted k-mers.
- Related methods (described in [THEORY.md](THEORY.md) §7): Bloom-filter inexact
  membership; BFC (BWT-based correction); de Bruijn graph compaction; POA/MSA
  consensus for long reads; EM for error-model learning.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/sequence-error-correction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/sequence-error-correction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\sequence-error-correction.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, prints the result, shows the
GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/reads_sample.txt` — 120 **synthetic** reads
  (60 bp, ~2% substitution noise) sampled from a random 400 bp genome at ~18×
  coverage, plus the error-free truth for each read so the demo can score itself.
  Runs offline, zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print how to obtain real
  benchmark reads (GAGE, GIAB, SRA) and convert them to the simple text format;
  they download nothing automatically and never bypass registration.
- **Regenerate / scale:** `python scripts/make_synthetic.py --reads 200000`.
- **Provenance & license:** see [data/README.md](data/README.md). The sample is
  synthetic and labelled as such.

Catalog dataset notes: GAGE short-read datasets (http://gage.cbcb.umd.edu/); GIAB
HG001–HG007 truth sets (https://www.nist.gov/programs-projects/genome-bottle);
ONT and PacBio CLR reads in the SRA (https://www.ncbi.nlm.nih.gov/sra).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt):

```
3.16 -- Sequence Error Correction
k-mer spectrum error correction (k=9, trust T=3)
reads: 120   total bases: 7200
spectrum: 1247 distinct 9-mers observed
corrections applied: 99 base(s) over 120 read(s)
errors vs truth:  before = 132   after = 39   (removed 93)
verify: spectrum_mismatch=0  corrected_mismatch=0
RESULT: PASS (GPU matches CPU exactly: spectrum + corrected reads)
```

The program builds the spectrum and corrects the reads on **both** the GPU
(`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`) and asserts
they are **byte-identical** — `spectrum_mismatch=0` and `corrected_mismatch=0`.
Because every operation is integer/byte work, the check is **exact** (no
floating-point tolerance; see [THEORY.md](THEORY.md) §6).

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads reads, runs CPU + GPU, verifies, reports.
2. [`src/reference_cpu.h`](src/reference_cpu.h) — the **shared `__host__ __device__`
   physics**: k-mer encoding, the trusted-k-mer predicate, and
   `correct_one_read()` (the one routine both CPU and GPU call).
3. [`src/kernels.cuh`](src/kernels.cuh) — the two-phase GPU interface + thread map.
4. [`src/kernels.cu`](src/kernels.cu) — `count_kmers_kernel` (atomic histogram) and
   `correct_reads_kernel` (one thread per read) + the host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline +
   the dataset loader.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **[CARE](https://github.com/fkallen/CARE)** — CUDA short-read error corrector;
  study how it keeps a k-mer **hash table** in GPU memory (the scalable
  replacement for our direct-indexed table) and uses minhashing to find
  overlapping reads.
- **[racon-GPU](https://github.com/NVIDIA-Genomics-Research/racon-gpu)** — GPU
  partial-order alignment (POA); study the *long-read* correction regime, where
  consensus from overlapping reads replaces the k-mer spectrum.
- **[CONSENT](https://github.com/morispi/CONSENT)** — long-read self-correction via
  local de Bruijn graphs; study how segmentation tames long, error-rich reads.
- **[Medaka](https://github.com/nanoporetech/medaka)** — RNN consensus for ONT;
  study how a learned error model (GPU inference) supersedes hand-set thresholds.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Two embarrassingly-parallel passes over reads. **Phase 1** is a parallel
**histogram**: one thread per read does `atomicAdd` into a shared count table —
integer atomics commute, so the result is deterministic and matches the CPU
exactly. **Phase 2** is the **"N independent jobs"** pattern (PATTERNS.md §1, as in
flagship `1.12`): one thread per read runs the shared `__host__ __device__`
correction routine against the read-only spectrum. The shared-core idiom
(PATTERNS.md §2) is what makes GPU == CPU *exact*.

## Exercises

1. **Sweep the trust threshold T** (in `main.cu`, `TRUST_THRESHOLD`). Too low and
   error k-mers get trusted (under-correction); too high and true k-mers get
   distrusted (over-correction → new errors). Plot errors-after vs T.
2. **Privatize the histogram.** Replace the global `atomicAdd` in
   `count_kmers_kernel` with a per-block shared-memory sub-histogram that is
   merged once at the end. Measure the speed-up at scale (`--reads 1000000`).
3. **Correct from both ends.** The current single left-to-right pass misses errors
   in the last *k*−1 bases (no k-mer starts there). Add a right-to-left pass (or
   correct using the k-mer *ending* at each position) and watch `errors after` drop.
4. **Two-bit pack the reads.** Store bases as 2 bits each in `uint32` words and
   rewrite `kmer_code_at` to slide with shifts — 4× less memory traffic, the path
   real tools take.
5. **Switch to a hash table.** Replace the 4^k direct table with an open-addressing
   GPU hash table (atomic CAS) so you can raise *k* to 15–21 (real coverage
   regimes) without a 4^k explosion.

## Limitations & honesty

- **Teaching-scope corrector.** A single left-to-right pass that fixes one
  substitution per k-mer start. It does **not** handle indels, does not iterate to
  convergence, and leaves errors in the last *k*−1 bases of each read and in dense
  error clusters — hence 39 of 132 errors remain in the demo. Production tools
  (CARE/BFC) iterate, correct from both directions, and model quality scores.
- **Direct-indexed spectrum.** We use *k* = 9 so the exact 4^9-slot table fits in
  ~1 MB (no hashing → perfectly legible). Real correctors use *k* = 15–31, which
  is impossible to direct-index (4^31 slots), so they use GPU **hash tables**
  (THEORY §7).
- **Synthetic data.** The reads are generated, not sequenced, with a clean
  substitution-only error model. Real reads have position-dependent error rates,
  indels, and biased motifs. The data is labelled synthetic everywhere.
- **Not a clinical tool.** Nothing here may inform diagnosis or treatment.
