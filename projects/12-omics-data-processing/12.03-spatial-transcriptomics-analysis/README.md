# 12.3 — Spatial Transcriptomics Analysis

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Analytical%20%26%20Omics%20Data%20Processing-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 12: Analytical & Omics Data Processing · Catalog ID `12.3`
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

Spatial transcriptomics (10x Visium, MERFISH, Xenium) measures gene expression at spatially defined locations (thousands of spots or millions of FISH-resolved single cells), producing large dense expression × spatial matrices. GPU acceleration applies to: (1) image-based spot detection and signal decoding for MERFISH (GPU-accelerated FISH barcode decoding), (2) dimension reduction and clustering (GPU UMAP / Leiden), and (3) spatial autocorrelation statistics (Moran's I computed as a sparse matrix-vector product over spatial neighbours). A 2025 biorxiv preprint describes GPU-accelerated 3D multiplexed iterative RNA-FISH decoding, and rctd-py delivers 9–41× GPU speedup for cell-type deconvolution of Visium HD (~400 k spots).

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

FISH barcode decoding (minimum-Hamming-distance matching, GPU parallel); spatial KNN graph construction; Moran's I spatial autocorrelation (sparse MVM); NMF/NNLS for deconvolution; SpatialDE spatially variable gene regression; GPU UMAP for spatial embedding.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/spatial-transcriptomics-analysis.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/spatial-transcriptomics-analysis.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\spatial-transcriptomics-analysis.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: 10x Genomics public spatial datasets — Visium/VisiumHD human tissue (https://www.10xgenomics.com/resources/datasets); Allen Brain Cell Atlas — spatial transcriptomics of whole mouse brain (https://portal.brain-map.org/atlases-and-data/bkp/abc-atlas); 4DN spatial data portal (https://data.4dnucleome.org/); MERSCOPE (Vizgen) public datasets (https://vizgen.com/data-release-program/).

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

rctd-py (https://github.com/p-gueguen/rctd-py) — GPU-accelerated RCTD deconvolution, 9–41× speedup; rapids-singlecell + Squidpy integration (https://github.com/scverse/rapids_singlecell) — GPU spatial analysis; GPU-accelerated RNA-FISH decoding (https://www.biorxiv.org/content/10.1101/2025.10.10.681751.full.pdf) — 3D FISH GPU processing; Squidpy (https://github.com/scverse/squidpy) — spatial omics analysis toolkit (GPU extension via rapids-singlecell).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuML UMAP / KNN for spatial graphs; cuSPARSE for spatial autocorrelation (Moran's I sparse MVM); cuDNN for FISH image decoding CNN; batched minimum-Hamming-distance kernels for MERFISH barcode matching; GPU tensor for dense spot × gene expression matrix. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
