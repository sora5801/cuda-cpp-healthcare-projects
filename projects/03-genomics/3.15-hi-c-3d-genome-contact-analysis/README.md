# 3.15 — Hi-C / 3D Genome Contact Analysis

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.15`
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

Hi-C maps chromatin contacts genome-wide, producing sparse contact matrices of size (genome_bins × genome_bins) at 1–10 kb resolution. Downstream analysis—matrix normalisation (ICE/KR balancing), TAD boundary calling, compartment A/B classification, and loop detection—involves iterative matrix operations on matrices with 3×10⁶ bins (3 Gb of data at 1 kb). GPU acceleration of the ICE iterative correction algorithm (repeated sparse matrix-vector products) and the 2D convolution-based loop caller (HiCCUPS) is particularly impactful. ChromaFold (2024) trains a lightweight CNN on a GPU to predict 3D contact maps from 1D accessibility signals.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

ICE / KR iterative matrix balancing (sparse MVM); eigendecomposition for A/B compartments; 1D insulation score for TAD boundary detection; HiCCUPS 2D Gaussian peak calling; Donut kernel convolution for loop enrichment; 3D polymer simulation constrained by Hi-C.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/hi-c-3d-genome-contact-analysis.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/hi-c-3d-genome-contact-analysis.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\hi-c-3d-genome-contact-analysis.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: 4DN (4D Nucleome) Data Portal — Hi-C across cell types and time (https://data.4dnucleome.org/); ENCODE Hi-C datasets — cell-line 3D contacts (https://www.encodeproject.org/); GEO Hi-C studies (GSE63525 Rao 2014 etc.) (https://www.ncbi.nlm.nih.gov/geo/); OpenChromatin Consortium ATAC/Hi-C (https://www.ncbi.nlm.nih.gov/geo/).

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

Higashi (https://github.com/ma-compbio/Higashi) — single-cell Hi-C GPU-accelerated hypergraph model; HiCCUPS (part of Juicer, https://github.com/aidenlab/juicer) — GPU-accelerated loop caller; ChromaFold (https://www.nature.com/articles/s41467-024-53628-0) — GPU CNN for contact prediction; cooler (https://github.com/open2c/cooler) — cool format Hi-C I/O (CPU, GPU matrix ops as next step).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuSPARSE for sparse ICE/KR matrix balancing; cuBLAS for dense compartment eigendecomposition; custom 2D convolution kernels (HiCCUPS); cuDNN for CNN-based contact prediction; GPU-resident contact matrix as CSR/CSC sparse format. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
