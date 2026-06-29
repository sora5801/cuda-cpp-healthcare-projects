# 3.25 — Base Quality Score Recalibration (BQSR)

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟢 Beginner · Established** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.25`
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

BQSR models and corrects systematic machine errors in Illumina base quality scores by regressing quality on covariates: read group, cycle position, sequence context (dinucleotide), and current reported quality. It requires scanning every base of every read (~1 trillion bases for a population study) against a known-variants database, computing covariate tables, then recalibrating scores. NVIDIA Parabricks GPU BQSR reimplements GATK's BaseRecalibrator in CUDA, processing a 30× WGS BQSR step in ~6 minutes on a DGX system vs. 4–9 hours on CPU, by parallelising covariate collection across reads in GPU thread blocks.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Log-linear regression over quality covariates; covariate table accumulation (parallel prefix sums); known-variant masking via hash look-up; empirical quality recalibration via quantised count table; dbSNP interval tree querying.

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
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, prints the result, shows the
GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/` — a tiny, offline input so the demo runs
  with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` (documented, idempotent).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: dbSNP build 155 — known variant positions for masking (https://www.ncbi.nlm.nih.gov/snp/); GiaB known-variant VCFs (https://www.nist.gov/programs-projects/genome-bottle); Mills and 1000G indels — GATK bundle known indels (https://storage.googleapis.com/genomics-public-data/); 1000 Genomes high-coverage WGS (https://www.internationalgenome.org/data).

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

NVIDIA Parabricks BQSR (https://docs.nvidia.com/clara/parabricks/latest/documentation/tooldocs/man_bqsr.html) — GPU BQSR, GATK-identical output; GATK4 BaseRecalibrator (https://github.com/broadinstitute/gatk) — CPU reference implementation; DeepVariant (https://github.com/google/deepvariant) — alternative CNN caller that bypasses BQSR need; Parabricks fq2bam (https://docs.nvidia.com/clara/parabricks/latest/documentation/tooldocs/man_fq2bam.html) — integrated BWA+BQSR+dedup pipeline.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Parallel covariate table reduction via atomicAdd; GPU hash table for known-variant look-up; shared-memory read buffers; cuBLAS for regression solve; one CUDA thread block per read batch; CUDA streams for pipelined I/O. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
