# 12.12 — Spatial Deconvolution of Cell Types

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Analytical%20%26%20Omics%20Data%20Processing-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 12: Analytical & Omics Data Processing · Catalog ID `12.12`
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

Spatial transcriptomics spots (Visium: 55 µm, ~10 cells/spot; Visium HD: 8 µm, ~1 cell) contain mixed gene expression signals from multiple cell types; deconvolution estimates cell-type proportions per spot using a scRNA-seq reference. RCTD (Robust Cell-Type Decomposition) fits a Poisson regression per spot independently—embarrassingly parallel—enabling GPU acceleration. rctd-py achieves 9–14× GPU speedup in doublet mode and 41× in multi-cell mode on VisiumHD (~400 k spots processed in ~1 minute on a Blackwell GPU). Cell2Location uses a hierarchical Bayesian model (pyro/PyTorch) with GPU MCMC; Tangram uses optimal transport GPU acceleration.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Poisson regression per spot (RCTD); negative binomial regression (Cell2Location); optimal transport spot-to-reference matching (Tangram); NMF for reference-free deconvolution; stereoscope GLM; spot clustering with GPU Leiden.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/spatial-deconvolution-of-cell-types.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/spatial-deconvolution-of-cell-types.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\spatial-deconvolution-of-cell-types.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: 10x Genomics Visium Human Tissue datasets (https://www.10xgenomics.com/resources/datasets); Allen Brain Cell Atlas spatial data (https://portal.brain-map.org/atlases-and-data/bkp/abc-atlas); MERSCOPE public data (https://vizgen.com/data-release-program/); 4DN spatial data (https://data.4dnucleome.org/).

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

rctd-py (https://github.com/p-gueguen/rctd-py) — GPU-accelerated RCTD, PyTorch backend; Cell2Location (https://github.com/BayraktarLab/cell2location) — hierarchical Bayesian GPU deconvolution; Tangram (https://github.com/broadinstitute/Tangram) — OT-based GPU spatial mapping; Squidpy (https://github.com/scverse/squidpy) — spatial analysis toolkit with rapids-singlecell integration.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Batched Poisson regression CUDA kernels (one CUDA block per spot); PyTorch CUDA for Bayesian MCMC; GPU optimal transport (POT/GeomLoss); cuML for reference PCA; multi-GPU Dask for VisiumHD-scale spot counts. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
