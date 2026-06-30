# 3.25 — Base Quality Score Recalibration (BQSR)

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟢 Beginner · Established** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.25`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Every base a sequencer calls comes with a **PHRED quality score** `Q` — the
machine's own estimate that the base is wrong (`P_err = 10^(-Q/10)`). Those scores
are **systematically biased**: the true error rate drifts with the machine cycle,
the local sequence context, and the reported `Q` itself. **Base Quality Score
Recalibration (BQSR)** measures that bias empirically — by scanning every base,
**masking known variants**, and tallying errors in a **covariate table** — then
rewrites every quality score to match the error rate actually observed. This
project builds that covariate table on the GPU with the **parallel-assign +
atomic integer reduction** pattern (one thread per base, `atomicAdd` into integer
bins), reproducing the CPU result **exactly**.

## What this computes & why the GPU helps

BQSR models and corrects systematic machine errors in Illumina base quality scores
by binning bases on covariates: cycle position, sequence context (di-nucleotide),
and current reported quality. It scans every base of every read (~1 trillion bases
for a population study) against a known-variants database, builds covariate count
tables, then recalibrates scores. NVIDIA Parabricks reimplements GATK's
`BaseRecalibrator` in CUDA, cutting a 30× WGS BQSR step from 4–9 CPU-hours to ~6
minutes by parallelising covariate collection across reads on the GPU.

**The parallel bottleneck:** the **covariate scan** — touching every base, masking
known sites, deciding error/no-error, and incrementing a bin counter. Each base is
independent until the tally, so we give each base its own GPU thread; the tally is
an `atomicAdd` scatter-reduction into a small shared table.

## The algorithm in brief

- **Mask:** skip bases at **known-variant** positions (dbSNP/Mills) — a mismatch
  there is biology, not machine error.
- **Bin:** for each surviving base, increment `obs` (and `err` if it mismatches
  the reference) in the covariate bin `(reported-Q, cycle, di-nucleotide context)`.
- **Empirical quality:** per bin, `Q_emp = -10·log10((err+1)/(obs+1))` (the +1 is
  GATK's Yates correction so a zero-error bin is finite).
- **Recalibrate:** each base's new quality = its bin's `Q_emp`.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/base-quality-score-recalibration-bqsr.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/base-quality-score-recalibration-bqsr.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\base-quality-score-recalibration-bqsr.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (CMake build)
```

The demo builds if needed, runs on `data/sample/`, prints the per-Q recalibration
summary, and verifies the GPU table + recalibrated qualities match the CPU
reference **exactly** (integer atomics commute → zero mismatches).

## Data

- **Sample (committed):** `data/sample/bqsr_sample.txt` — a tiny **synthetic**
  alignment (1200 reads × 12 bp over a 24 bp reference, 2 known-variant sites) so
  the demo runs with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print pointers (real BQSR
  reads BAM + dbSNP/Mills VCFs — see below).
- **Provenance & license:** see [data/README.md](data/README.md).
- Bigger synthetic set: `python scripts/make_synthetic.py --reads 50000`.

Catalog dataset notes: dbSNP build 155 — known variant positions for masking
(<https://www.ncbi.nlm.nih.gov/snp/>); GiaB known-variant VCFs
(<https://www.nist.gov/programs-projects/genome-bottle>); Mills and 1000G indels —
GATK bundle (<https://storage.googleapis.com/genomics-public-data/>); 1000 Genomes
high-coverage WGS (<https://www.internationalgenome.org/data>).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The GPU
(`src/kernels.cu`) and CPU (`src/reference_cpu.cpp`) share the covariate model and
PHRED math (`src/bqsr.h`), so the **integer covariate table** and the
**recalibrated qualities** are **bit-identical** (`table mismatches = 0`,
`quality mismatches = 0`). The headline line — `Q=30 … -> Q_emp=19` — is BQSR's
whole point: the reads were *reported* at Q30, but the bases actually erred at
roughly Q19, and recalibration fixes that. The two known-variant columns are masked
out, so they do **not** inflate the error count.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the alignment, runs CPU + GPU, verifies, reports.
2. [`src/bqsr.h`](src/bqsr.h) — the **shared (host+device) covariate model**: bin
   indexing, `classify_base`, PHRED ↔ probability, and `empirical_q`.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + thread-mapping idea.
4. [`src/kernels.cu`](src/kernels.cu) — `accumulate_kernel` (**atomic** table build)
   + `recalibrate_kernel` + the host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the loader + trusted serial baseline.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **NVIDIA Parabricks BQSR** (<https://docs.nvidia.com/clara/parabricks/latest/documentation/tooldocs/man_bqsr.html>)
  — GPU BQSR with GATK-identical output; the production version of this idea.
- **GATK4 `BaseRecalibrator`** (<https://github.com/broadinstitute/gatk>) — the
  canonical CPU reference; study its covariate set and Yates correction.
- **DeepVariant** (<https://github.com/google/deepvariant>) — a CNN caller that
  learns base-quality context internally and so can *skip* BQSR.
- **Parabricks `fq2bam`** (<https://docs.nvidia.com/clara/parabricks/latest/documentation/tooldocs/man_fq2bam.html>)
  — the integrated BWA + dedup + BQSR pipeline BQSR slots into.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Parallel assign + atomic integer reduction** (PATTERNS.md, exemplified by
flagship 11.09): one CUDA thread per base, `atomicAdd` into integer covariate-table
bins. Integer counters make the atomics **commute** → the table is deterministic
and matches the CPU exactly. Known-variant masking is a per-base array lookup; the
empirical-quality solve is a closed-form PHRED of the count ratio (no library
needed at this scale — see Exercise 5 and THEORY §7 for the regression view).

## Exercises

1. **Watch masking matter.** In `scripts/make_synthetic.py`, blank the `KNOWN`
   line (no masked sites) and rebuild the sample. The known-variant columns now
   count as machine errors, the error tally jumps, and `Q_emp` drops — a vivid
   demonstration of *why* BQSR masks known variants.
2. **Per-cycle covariate.** The table already keys on cycle; print `Q_emp` per
   cycle (not just aggregated) and inject a cycle-dependent error rate in the
   generator to see the end-of-read quality droop real sequencers exhibit.
3. **Float vs integer atomics.** Replace the integer `atomicAdd` counters with
   floats and observe the table (and thus `Q_emp`) wobble run-to-run — the
   determinism lesson (CLAUDE.md §12, PATTERNS.md §3).
4. **Shared-memory table.** For one read group the table is small; cache the
   per-block partial counts in shared memory and reduce once per block to cut
   global-atomic traffic.
5. **The regression view.** GATK fits a hierarchical log-linear model over the
   covariates rather than independent bins. Replace the per-bin empirical quality
   with a global Q offset plus per-covariate deltas solved by least squares
   (cuBLAS/cuSOLVER) and compare.

## Limitations & honesty

- **Reduced-scope teaching version.** We model a **single read group** with a
  small `(Q, cycle, di-nucleotide)` table; production BQSR adds read group, more
  context length, indel covariates, and a hierarchical **log-linear regression**
  (not independent per-bin estimates). The pattern and the masking are faithful;
  the model is simplified (THEORY §7).
- **Synthetic data.** `data/sample/` is generated Gaussian-free synthetic reads
  with a *known* injected error rate so the recovered `Q_emp` is interpretable. It
  is **not** real sequencing data and carries no clinical meaning.
- The recalibrated scores here are for learning the covariate-table mechanics, not
  for feeding a real variant caller.
