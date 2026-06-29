# 12.11 — Trajectory Inference & Pseudotime Analysis

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Analytical%20%26%20Omics%20Data%20Processing-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 12: Analytical & Omics Data Processing · Catalog ID `12.11`
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

Trajectory inference reconstructs continuous developmental processes from snapshot scRNA-seq data by ordering cells along a pseudotime axis representing biological progression (differentiation, cell cycle, immune activation). Algorithms range from principal curve fitting (Monocle3) to diffusion-map graph-based approaches (Scanpy PAGA) and optimal transport (Waddington-OT). GPU acceleration targets the KNN graph construction (the shared first step), the diffusion map eigensolver (cuSolver), and the optimal transport (Sinkhorn algorithm) computation. For atlas-scale data (>1 M cells), GPU trajectory inference with RAPIDS reduces hours to minutes.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Principal curve / elastic principal graph (DDRTree); diffusion pseudotime (DPT) via diffusion map eigenvectors; PAGA graph abstraction; RNA velocity (scVelo) splicing dynamics EM; Sinkhorn optimal transport for fate probability; graph-based geodesic distances for branch assignment.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/trajectory-inference-pseudotime-analysis.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/trajectory-inference-pseudotime-analysis.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\trajectory-inference-pseudotime-analysis.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Human Cell Atlas developmental atlases (https://www.humancellatlas.org/); GEO scRNA-seq differentiation time-course datasets (https://www.ncbi.nlm.nih.gov/geo/); Allen Brain Cell Atlas (https://portal.brain-map.org/atlases-and-data/bkp/abc-atlas); ENCODE iPSC differentiation scRNA-seq (https://www.encodeproject.org/).

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

rapids-singlecell (https://github.com/scverse/rapids_singlecell) — GPU diffusion pseudotime and UMAP; Scanpy with GPU backend (https://github.com/scverse/scanpy) — PAGA trajectory analysis; scVelo (https://github.com/theislab/scvelo) — RNA velocity (GPU EM target); Monocle3 (https://github.com/cole-trapnell-lab/monocle3) — principal graph trajectory inference.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuSolver eigenvalue solver for diffusion map; cuSPARSE for KNN graph Laplacian operations; cuML for PCA pre-reduction; custom Sinkhorn CUDA kernels (iterative row/column normalisation); GPU optimal transport via POT library with CUDA backend. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
