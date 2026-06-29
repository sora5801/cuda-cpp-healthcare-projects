# 7.7 — Multi-Omics Integration

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20AI%20%26%20Clinical%20Deep%20Learning-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 7: Medical AI & Clinical Deep Learning · Catalog ID `7.7`
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

Combines heterogeneous molecular data layers — genomics (SNP/CNV), transcriptomics (RNA-seq), proteomics, metabolomics, and epigenomics — to predict disease subtype, drug response, or patient outcome. Integrating these layers requires jointly embedding high-dimensional sparse matrices (gene expression: 20k genes × 10k patients) with dense low-dimensional clinical vectors. GPUs accelerate the large embedding layers and transformer attention that learn cross-modal correspondences; a single multi-omics autoencoder can have hundreds of millions of parameters when modelling all layers simultaneously. scGPT-style tokenisation of omics measurements treats genes as tokens and uses CUDA-accelerated attention. Sparse input matrices benefit from cuSPARSE SpMM operations.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Multi-modal autoencoders (VAE, VQVAE), Graph Neural Networks over molecular interaction networks, transformer tokenisation (scGPT, mosGraphGPT), MOFA+ factor analysis, multi-task learning across omics, contrastive multi-omics pre-training, pathway-guided sparse attention.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/multi-omics-integration.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/multi-omics-integration.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\multi-omics-integration.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: TCGA Pan-Cancer Atlas — genomic, transcriptomic, proteomic data for 33 cancer types (https://www.cancer.gov/tcga) GEO (Gene Expression Omnibus) — 5M+ omics samples across species/conditions (https://www.ncbi.nlm.nih.gov/geo/) CPTAC (Clinical Proteomic Tumor Analysis Consortium) — proteogenomics across tumour types (https://proteomics.cancer.gov/programs/cptac) ENCODE — chromatin, transcription factor, and RNA datasets (https://www.encodeproject.org/)

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

scGPT (https://github.com/bowang-lab/scGPT) — GPT-style multi-omics foundation model with GPU pretraining MOFA+ (https://github.com/bioFAM/MOFA2) — factor analysis for multi-omics (CPU; GPU via JAX backend) TF-DWGNet (https://arxiv.org/abs/2509.16301) — directed weighted GNN for multi-omics cancer subtype classification (verify URL) MOLI / Concrete Autoencoder (https://github.com/mims-harvard/Madrigal) — multi-omics latent integration (verify URL)

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuSPARSE for sparse omics matrices, Flash Attention for gene-token sequences, NCCL multi-GPU; pattern: column-parallel embedding for gene dimension, row-parallel for sample dimension. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
