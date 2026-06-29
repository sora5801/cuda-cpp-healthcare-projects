# 3.3 — Variant Calling Acceleration

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟢 Beginner · Established** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.3`
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

Germline variant calling applies the Haplotype Caller algorithm: local de novo assembly of active regions, PairHMM forward-algorithm computation of read-haplotype likelihoods, and genotype likelihood calculation. PairHMM is by far the dominant runtime cost—each read must be compared against every candidate haplotype via an O(R×H) DP table. GPU parallelism fills an entire PairHMM table per thread block, running thousands of read-haplotype pairs simultaneously. Parabricks GPU HaplotypeCaller reduces 30× WGS germline calling from ~9 hours CPU to under 10 minutes on an H100 using GATK-identical math. DeepVariant's CNN pileup scoring is a further candidate for batched GPU inference.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

PairHMM forward algorithm; local de novo assembly (De Bruijn graph over active regions); Viterbi realignment; genotype likelihood calculation (GL/PL); base quality score recalibration (BQSR); DeepVariant convolutional inference.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
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

Catalog dataset notes: GiaB truth sets HG001–HG007 — gold-standard variant calls for benchmarking (https://www.nist.gov/programs-projects/genome-bottle); ClinVar — clinically interpreted variants (https://www.ncbi.nlm.nih.gov/clinvar/); gnomAD v4 — population allele frequencies (https://gnomad.broadinstitute.org/); 1000 Genomes high-coverage WGS (https://www.internationalgenome.org/data).

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

NVIDIA Parabricks HaplotypeCaller / DeepVariant module (https://docs.nvidia.com/clara/parabricks/latest/) — GATK-identical GPU variant calling; DeepVariant (https://github.com/google/deepvariant) — CNN-based caller deployable on GPU; GATK (https://github.com/broadinstitute/gatk) — CPU reference for parity testing; Clairvoyante / Clair3 (https://github.com/HKU-BAL/Clair3) — deep learning variant caller with GPU inference.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuDNN (DeepVariant CNN inference); custom PairHMM CUDA kernels with one block per read-haplotype pair; shared-memory DP tables; multi-GPU pipeline parallelism (BQSR → alignment → calling); CUDA streams for pipelining I/O and compute. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
