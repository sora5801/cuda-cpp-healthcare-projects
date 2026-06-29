# 12.9 — GPU UMAP / t-SNE for Single-Cell Omics

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Analytical%20%26%20Omics%20Data%20Processing-lightgrey)

> **🟢 Beginner · Established** — Domain 12: Analytical & Omics Data Processing · Catalog ID `12.9`
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

UMAP and t-SNE dimensionality reduction are the universal visualisation steps in single-cell omics (scRNA-seq, scATAC-seq, CyTOF, CITE-seq). For a million-cell dataset, standard CPU UMAP takes hours; GPU UMAP (cuML) and GPU t-SNE (RAPIDS) reduce this to minutes by parallelising the KNN graph construction (Faiss-GPU approximate nearest neighbours) and the repulsive/attractive force optimisation (each cell's gradient update is independent given the current embedding). The NVIDIA blog demonstrates GPU UMAP on 1.3 M cells processing in ~1 minute vs. ~40 minutes CPU.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Exact/approximate KNN (Faiss IVF-PQ, HNSWLIB-GPU); fuzzy simplicial set construction (UMAP); stochastic gradient descent with negative sampling (UMAP layout); t-SNE Barnes-Hut or FIt-SNE approximation; PCA for pre-reduction; Leiden/Louvain graph clustering.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/gpu-umap-t-sne-for-single-cell-omics.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/gpu-umap-t-sne-for-single-cell-omics.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\gpu-umap-t-sne-for-single-cell-omics.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Human Cell Atlas 10x datasets (https://www.humancellatlas.org/); CellxGene Census — 50 M+ cells (https://cellxgene.cziscience.com/); 10x Genomics 1.3 M mouse brain dataset (https://www.10xgenomics.com/resources/datasets); NCBI GEO scRNA-seq compendium (https://www.ncbi.nlm.nih.gov/geo/).

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

rapids-singlecell (https://github.com/scverse/rapids_singlecell) — GPU UMAP/Leiden/PCA for scRNA-seq; cuML (https://github.com/rapidsai/cuml) — GPU UMAP and t-SNE via RAPIDS; Faiss (https://github.com/facebookresearch/faiss) — GPU KNN for UMAP graph construction; NVIDIA RAPIDS single-cell examples (https://github.com/NVIDIA-Genomics-Research/rapids-single-cell-examples) — benchmarked notebooks.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuML UMAP (GPU KNN + SGD layout); Faiss-GPU IVF-PQ approximate nearest neighbours; cuGraph for Leiden clustering; CUB warp-level reduction for gradient accumulation; atomic updates for asynchronous UMAP layout; multi-GPU via Dask for >10 M cells. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
