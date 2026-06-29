# 11.9 — Flow Cytometry & High-Content Screening Analysis

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Biotechnology%2C%20Bioprocess%20%26%20Synthetic%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 11: Biotechnology, Bioprocess & Synthetic Biology · Catalog ID `11.9`
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

Modern cell sorters generate 10⁶ cells/second at 20–50 parameters per event; high-content screening (HCS) platforms image millions of cells per plate with 10+ channels. GPU-accelerated dimensionality reduction (GPU-UMAP, GPU-TSNE via RAPIDS cuML) and clustering (GPU-HDBSCAN, GPU-PhenoGraph) turn 30-minute analyses into seconds, enabling real-time sort gates. GPU-accelerated CellProfiler-style morphological feature extraction processes 96-well plate images in minutes instead of hours. Deep-learning classifiers (ResNet, ViT) deployed on GPU identify rare phenotypes (1-in-10⁵ events) with high sensitivity.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

GPU-UMAP (approximate nearest-neighbor with NN-descent), GPU-HDBSCAN, GPU FlowSOM self-organizing map, GPU PhenoGraph graph-based clustering, GPU CellPose segmentation for HCS, Wasserstein distance for batch-effect correction, GPU deep learning rare-event classifier.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/flow-cytometry-high-content-screening-analysis.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/flow-cytometry-high-content-screening-analysis.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\flow-cytometry-high-content-screening-analysis.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: FlowRepository — public flow cytometry FCS files (https://flowrepository.org/); JUMP-CP — 116 K compound HCS morphological profiles, RxRx cell-painting images (https://jump-cellpainting.broadinstitute.org/); Cell Painting Gallery (Broad Institute) — 140 TB cell images (https://registry.opendata.aws/cellpainting-gallery/); Human Protein Atlas imaging (https://www.proteinatlas.org/).

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

RAPIDS cuML (https://github.com/rapidsai/cuml) — GPU UMAP/TSNE/HDBSCAN for cytometry analysis; CellProfiler (https://github.com/CellProfiler/CellProfiler) — HCS morphological profiling (with GPU CellPose segmentation); CellPose (https://github.com/mouseland/cellpose) — GPU-accelerated cell segmentation; FlowKit (https://github.com/whitews/FlowKit) — FCS file processing (CPU; upstream of GPU analysis).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuML GPU-UMAP, cuDNN for ResNet cell image classifier, CUDA 2D convolution kernels for morphological feature extraction; pattern: FCS/image batch ingest → GPU feature extraction → GPU-UMAP embedding → GPU-HDBSCAN clustering → rare-event gating → real-time sort decisions. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
