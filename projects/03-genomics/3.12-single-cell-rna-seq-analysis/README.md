# 3.12 — Single-Cell RNA-seq Analysis

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Genomics%2C%20Sequencing%20%26%20Bioinformatics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 3: Genomics, Sequencing & Bioinformatics · Catalog ID `3.12`
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

Single-cell RNA-seq (scRNA-seq) produces count matrices for tens of millions of cells × 30 k genes; downstream analysis involves normalisation, highly variable gene selection, PCA, k-nearest-neighbour graph construction (O(n²) naive, accelerated by approximate nearest neighbours), UMAP / t-SNE embedding, Leiden/Louvain clustering, and differential expression. rapids-singlecell (scverse, 2024) replaces Scanpy's NumPy/SciPy backend with cuPy, cuML, and cuGraph equivalents, achieving >20× speedup for datasets up to 20 M cells. The KNN graph construction and UMAP optimisation are the most GPU-impactful steps, turning hours into minutes.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Normalised count transformation (scran/Seurat); PCA on sparse count matrix; approximate KNN (Faiss, HNSWLIB GPU); UMAP force-directed layout; Leiden graph clustering; negative binomial GLM for differential expression; doublet detection.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/single-cell-rna-seq-analysis.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/single-cell-rna-seq-analysis.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\single-cell-rna-seq-analysis.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Human Cell Atlas — multi-organ scRNA-seq compendium (https://www.humancellatlas.org/); 10× Genomics public datasets (https://www.10xgenomics.com/resources/datasets); CellxGene Census — 50 M+ cells (https://cellxgene.cziscience.com/); NCBI GEO — thousands of scRNA-seq studies (https://www.ncbi.nlm.nih.gov/geo/).

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

rapids-singlecell (https://github.com/scverse/rapids_singlecell) — drop-in GPU Scanpy replacement, cuPy/cuML/cuGraph; NVIDIA RAPIDS single-cell examples (https://github.com/NVIDIA-Genomics-Research/rapids-single-cell-examples) — benchmark notebooks up to 1 M+ cells; ScaleSC (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12321287/) — GPU scRNA pipeline, 20× speed, 20 M cells on A100; Scanpy (https://github.com/scverse/scanpy) — CPU reference with GPU-aware backends.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuPy sparse GEMM (count matrix ops); cuML PCA, UMAP, KNN; cuGraph Leiden/Louvain; Faiss-GPU HNSW index for ANN; cuDF for dataframe operations; multi-GPU Dask for datasets exceeding GPU RAM. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
