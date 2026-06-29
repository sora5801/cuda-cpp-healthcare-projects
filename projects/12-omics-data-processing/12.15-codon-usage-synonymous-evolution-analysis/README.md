# 12.15 — Codon Usage & Synonymous Evolution Analysis

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Analytical%20%26%20Omics%20Data%20Processing-lightgrey)

> **🟢 Beginner · Established** — Domain 12: Analytical & Omics Data Processing · Catalog ID `12.15`
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

Codon usage analysis computes codon adaptation index (CAI), relative synonymous codon usage (RSCU), and dN/dS (non-synonymous to synonymous substitution ratio) across thousands of gene alignments, often in population genomics or viral evolution studies. dN/dS computation requires pairwise codon alignment followed by branch-model likelihood evaluation per codon triplet—a compute-intensive phylogenetic likelihood calculation. For genome-scale dN/dS scans (10⁶ gene pairs), GPU parallelism assigns one CUDA thread per gene pair, with codon frequency tables in shared memory. Combined with phylogenetic likelihood (Section 3.9 BeagleLib), GPU codon models enable population-scale selection scans.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Codon substitution model (Goldman-Yang GY94, MG94); dN/dS branch-site model likelihood; RSCU and CAI calculation; synonymous site rate estimation; Fisher's exact test for codon usage bias; maximum likelihood codon tree.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/codon-usage-synonymous-evolution-analysis.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/codon-usage-synonymous-evolution-analysis.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\codon-usage-synonymous-evolution-analysis.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Ensembl CDS sequences — comparative codon data across species (https://www.ensembl.org/); NCBI RefSeq CDS archives (https://ftp.ncbi.nlm.nih.gov/refseq/); GISAID SARS-CoV-2 genomes — viral codon evolution dataset (https://www.gisaid.org/); PopHuman dN/dS datasets (verify URL).

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

BeagleLib (https://github.com/beagle-dev/beagle-lib) — GPU codon model likelihood evaluation; HyPhy (https://github.com/veg/hyphy) — GPU-capable dN/dS and selection analysis framework; PAML (http://abacus.gene.ucl.ac.uk/software/paml.html) — CPU dN/dS reference; BEAST2 (https://github.com/CompEvol/beast2) — Bayesian molecular evolution using BeagleLib GPU.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

BeagleLib CUDA kernels for 60×60 codon matrix-vector products; one CUDA thread per alignment column per codon model; cuBLAS for codon substitution matrix exponentiation; GPU-resident codon frequency tables; multi-GPU tree partitioning. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
