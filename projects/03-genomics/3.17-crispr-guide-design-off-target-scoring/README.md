# 3.17 — CRISPR Guide Design & Off-Target Scoring

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.17`
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

Designing effective CRISPR guide RNAs requires genome-wide off-target assessment: every 20-mer protospacer must be compared against all near-matches in the genome (allowing mismatches and bulges). For a 3 Gb human genome, this is ~300 M potential off-target sites per guide; Cas-OFFinder uses GPU to enumerate all combinations of mismatches in parallel. Scoring each off-target for actual cutting probability requires a learned model (CFD score, CNN, transformer), which GPU inference accelerates in batch over all candidate sites. FlashFry precomputes a compressed binary index enabling fast GPU-scalable off-target database look-ups.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Exact/approximate string matching with bounded mismatches (BFS over mismatch graph); CFD (cutting frequency determination) scoring; CNN/RNN on-target efficiency prediction; protein language model (PLM) for Cas9 variant activity (PLM-CRISPR); off-target enumeration via FM-index or hash-based inexact search.

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

Catalog dataset notes: CRISPOR benchmark — validated guide efficiencies and off-targets (https://crispor.gi.ucsc.edu/); GeCKO v2 library — genome-scale CRISPR knockout screen guides (https://www.addgene.org/pooled-library/leczkowski-gecko-v2/); Azimuth / Rule Set 2 training data — published guide efficiency datasets (verify URL); hg38/mm10 reference genomes — for off-target genome scanning (https://genome.ucsc.edu/).

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

Cas-OFFinder (https://github.com/snugel/cas-offinder) — GPU-accelerated off-target search, mismatch + RNA bulge enumeration; FlashFry (https://github.com/aaronmck/FlashFry) — scalable CRISPR target design with binary index; CRISPOR (https://github.com/maximilianh/crisporPaper) — comprehensive on/off-target scoring pipeline; PLM-CRISPR (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12254127/) — protein LM for Cas9 variant activity prediction with GPU inference.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom CUDA mismatch enumeration kernels (parallel BFS across mismatch positions); GPU-resident genome index in constant/global memory; cuDNN for CNN on-target scoring; batched transformer inference (ESM / PLM) on GPU; one CUDA thread per genome position. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
