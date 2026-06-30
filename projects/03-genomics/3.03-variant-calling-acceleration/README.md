# 3.3 — Variant Calling Acceleration

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟢 Beginner · Established** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.3`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

This project implements the **PairHMM forward algorithm**, the computational heart
of germline variant calling (GATK HaplotypeCaller, NVIDIA Parabricks). Given a set
of sequencing **reads** and a set of candidate **haplotypes** (hypothesised local
genome sequences), it computes, for every (read, haplotype) pair, the likelihood
`P(read | haplotype)` under a three-state (Match/Insert/Delete) hidden Markov model
that accounts for sequencing error. Every pair is an independent dynamic-programming
table, so we give **one GPU thread per pair** and fill thousands of tables at once.
The demo loads a tiny synthetic locus, runs the PairHMM on both GPU and CPU, checks
they agree to machine precision, and assigns each read to its most-likely haplotype.

## What this computes & why the GPU helps

Germline variant calling applies the HaplotypeCaller algorithm: local de novo
assembly of active regions, **PairHMM forward-algorithm computation of
read-haplotype likelihoods**, and genotype-likelihood calculation. PairHMM is by
far the dominant runtime cost — each read must be compared against every candidate
haplotype via an O(R×H) DP table. GPU parallelism fills entire PairHMM tables in
parallel, running thousands of read-haplotype pairs simultaneously. Parabricks GPU
HaplotypeCaller reduces 30× whole-genome germline calling from ~9 hours on CPU to
under 10 minutes on a datacentre GPU using GATK-identical math.

**The parallel bottleneck:** scoring `n_reads × n_haps` read-haplotype pairs. Each
pair's `O(R·H)` forward DP table is **independent** of every other pair — no shared
state, no ordering — which is precisely why the workload maps onto a grid of
independent GPU threads. This project parallelizes exactly that step.

## The algorithm in brief

- **PairHMM forward algorithm** — sum over all alignment paths of a read to a
  haplotype through Match/Insert/Delete states; the marginal likelihood `P(r|h)`.
- **Phred emission model** — base error probability `e = 10^(-Q/10)`; match emits
  `1-e`, mismatch emits `e/3`.
- **Two-row DP** — keep only the previous and current rows, `O(H)` memory per pair.
- **Per-read argmax** — assign each read to its most-likely haplotype (the headline
  result).

The full pipeline (local assembly of haplotypes, BQSR, genotype likelihoods,
DeepVariant CNN scoring) is described — and our simplifications named — in
[THEORY.md](THEORY.md), which gives the science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/variant-calling-acceleration.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/variant-calling-acceleration.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\variant-calling-acceleration.sln /p:Configuration=Release /p:Platform=x64
```

Both `Release|x64` and `Debug|x64` build with zero warnings. Because the math is
shared `double` code (`src/pairhmm_core.h`), the two configs produce byte-identical
stdout.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, prints the per-read haplotype
assignments, shows the GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/reads_haplotypes_sample.txt` — a tiny,
  **synthetic**, offline input (1 truth haplotype + 2 alternatives, 8 reads drawn
  from the truth) so the demo runs with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print links and
  instructions (they never bypass registration). Regenerate or enlarge the
  synthetic input with `python scripts/make_synthetic.py`.
- **Provenance & license & file format:** see [data/README.md](data/README.md).

Catalog benchmark resources: GiaB truth sets HG001–HG007
(https://www.nist.gov/programs-projects/genome-bottle); ClinVar
(https://www.ncbi.nlm.nih.gov/clinvar/); gnomAD v4
(https://gnomad.broadinstitute.org/); 1000 Genomes high-coverage WGS
(https://www.internationalgenome.org/data).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): every
read is assigned to haplotype 0 (the truth), and `RESULT: PASS`. The program
computes the log-likelihood matrix on both the **GPU** (`src/kernels.cu`) and a
**CPU reference** (`src/reference_cpu.cpp`) and asserts they agree within
`1.0e-9`; the measured error is ~`1.8e-15` (machine precision) because both paths
run the identical `double` recurrence from `src/pairhmm_core.h`. That agreement,
plus the reads landing on the truth haplotype, is the correctness guarantee.

## Code tour

Read in this order:

1. [`src/pairhmm_core.h`](src/pairhmm_core.h) — the **shared** per-cell PairHMM math
   (`pairhmm_step`, `base_emission_prob`), `__host__ __device__` so CPU and GPU run
   identical arithmetic. Start here; it is the heart of the project.
2. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-pair
   mapping idea.
4. [`src/kernels.cu`](src/kernels.cu) — the `pairhmm_kernel` and its host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline +
   the data loader.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, I/O helpers.

## Prior art & further reading

- **NVIDIA Parabricks HaplotypeCaller / DeepVariant**
  (https://docs.nvidia.com/clara/parabricks/latest/) — GATK-identical GPU variant
  calling; study its one-block-per-pair, shared-memory PairHMM design.
- **GATK** (https://github.com/broadinstitute/gatk) — the CPU reference for parity
  testing; read the `pairhmm` package for the exact recurrence and row rescaling.
- **DeepVariant** (https://github.com/google/deepvariant) — CNN-pileup caller; the
  alternative where cuDNN inference replaces a DP kernel.
- **Clairvoyante / Clair3** (https://github.com/HKU-BAL/Clair3) — deep-learning
  caller with GPU inference, strong on long reads.

Study these to learn the production approach; **do not copy code wholesale** —
this project reimplements the math didactically and credits the sources
(CLAUDE.md §2).

## CUDA pattern used here

**Independent jobs** (PATTERNS.md §1): one CUDA thread per (read, haplotype) pair,
each filling its own forward DP table with two rolling rows in per-thread local
memory; no shared memory, no atomics, no inter-thread communication. The shared
`__host__ __device__` core (PATTERNS.md §2) makes the GPU and CPU results
bit-comparable. The production refinement — **one thread block per pair** with a
**shared-memory anti-diagonal** DP table (the wavefront of project 3.01) — is
explained in THEORY.md §4 and left as an exercise.

## Exercises

1. **Wavefront a single table.** Switch from one-thread-per-pair to
   one-block-per-pair, filling each DP table along anti-diagonals in shared memory
   (cf. project 3.01). Compare kernel time as `read_len` grows.
2. **Per-base gap penalties.** Replace the constant `δ`/`ε` with per-base
   insertion/deletion qualities (extend the data format and `PairHmmParams`), and
   see how the likelihoods shift.
3. **Row rescaling.** Add GATK-style per-row rescaling so 250-base reads don't
   underflow even before the final `log10`. Verify the result is unchanged within
   tolerance.
4. **Scale it.** Run `make_synthetic.py --reads 4096 --read-len 100 --hap-len 120`
   and watch the GPU overtake the CPU as the pair count grows (the timing is a
   teaching artifact — see THEORY.md §honest-timing).
5. **Add a fourth haplotype** with a small deletion and confirm reads with a
   matching deletion prefer it.

## Limitations & honesty

- **Synthetic data.** The committed sample is generated, labeled synthetic
  everywhere, and is **not** real or patient-derived. No output is a clinical
  result.
- **Reduced scope.** This implements only the PairHMM forward step. It does **not**
  do local assembly (the haplotypes are given), base-quality recalibration,
  genotype-likelihood/PL computation, joint genotyping, or VCF emission — the rest
  of a real caller. THEORY.md §7 names each omission.
- **Simplified model.** Constant gap-open/extend probabilities (not per-base
  quality-derived), uniform insertion emission, fixed-length reads/haplotypes, and
  a compile-time cap `hap_len ≤ 127` for the per-thread DP rows.
- **Timing is a teaching artifact, not a benchmark.** On the tiny sample the GPU is
  *slower* than the CPU (launch overhead dominates); the GPU's advantage appears at
  scale. We state the measured numbers honestly and never claim a speed-up here.
