# 14.8 — Real-Time Genomic Pathogen Surveillance Networks

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Emerging%2C%20Theoretical%20%26%20Grand--Challenge%20Frontiers-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 14: Emerging, Theoretical & Grand-Challenge Frontiers · Catalog ID `14.8`
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

Epidemic genomic surveillance sequences thousands of viral/bacterial isolates per day, requiring near-real-time genome assembly, variant calling, phylogenetic placement, and transmission cluster detection. GPU-accelerated genome assembly (GPU-MEGAHIT) and variant calling (GPU Parabricks) reduce per-sample analysis from hours to minutes, enabling next-flight sequencing decisions during outbreak response. Phylogenetics on GPU (iqtree GPU, PhyML-CUDA) computes maximum likelihood trees on thousands of taxa. Real-time cluster detection via GPU-accelerated pairwise SNP distance matrices (all-vs-all on N×N matrix) parallelizes naturally over GPU threads.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

GPU-accelerated de novo assembly (BWT-based, de Bruijn graph), GPU variant calling (Parabricks Haplotypecaller), maximum likelihood phylogenetics (GTR+Γ model), pairwise SNP distance matrix, Bayesian temporal phylogenetics (BEAST GPU backend), epidemic growth rate estimation (SEIR model on GPU).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/real-time-genomic-pathogen-surveillance-networks.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/real-time-genomic-pathogen-surveillance-networks.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\real-time-genomic-pathogen-surveillance-networks.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: GISAID EpiCoV — 17M+ SARS-CoV-2 genomes (https://gisaid.org/); NCBI SRA — all short-read sequencing submissions (https://www.ncbi.nlm.nih.gov/sra); Nextstrain builds — curated SARS-CoV-2 / influenza phylogenies (https://nextstrain.org/); PHA4GE pathogen genomics standards datasets (https://pha4ge.org/).

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

NVIDIA Clara Parabricks (https://www.nvidia.com/en-us/clara/parabricks/) — GPU genome assembly/variant calling (40× speedup over GATK); Nextstrain (https://github.com/nextstrain/ncov) — phylogenetic outbreak analysis pipeline; IQ-TREE (https://github.com/Cibiv/IQ-TREE) — ML phylogenetics (multi-GPU via CUDA); GPU-MEGAHIT (https://github.com/GPU-MEGAHIT/GPU-MEGAHIT) — GPU-accelerated metagenomics assembly.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CUDA BWT for GPU read alignment (BWA-MEM on CUDA), cuBLAS for SNP distance matrix computation, cuFFT for k-mer frequency analysis; pattern: raw reads → GPU assembly → GPU variant calling → pairwise SNP matrix on GPU → transmission cluster detection → phylogenetic placement → epidemiological alert. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
