# 2.6 — Normal Mode Analysis / Elastic Network Models

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟢 Beginner · Established** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.6`
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

Normal Mode Analysis (NMA) computes the low-frequency vibrational modes of a protein structure, revealing collective motions (domain movements, breathing modes) relevant to allostery and function. The bottleneck is diagonalization of the 3N×3N Hessian matrix (N = atom count) — an O(N³) dense eigenvalue problem. For large proteins (N > 50,000 atoms) this is intractable on CPU. Elastic Network Models (ENMs: ANM, GNM) use simplified Hookean springs between Cα atoms, reducing the matrix but still benefiting from GPU cuSOLVER for eigendecomposition and CUDA-accelerated matrix-vector products (Lanczos iteration for sparse NMA).

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Anisotropic network model (ANM), Gaussian network model (GNM), Hessian matrix construction (pairwise spring contacts), Lanczos/ARPACK for sparse eigendecomposition, overlap with experimental B-factors/conformational changes, RMSF from mode summation.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/normal-mode-analysis-elastic-network-models.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/normal-mode-analysis-elastic-network-models.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\normal-mode-analysis-elastic-network-models.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: PDB protein structures (https://www.rcsb.org); ProDy structural dynamics dataset (https://github.com/prody/ProDy); MoDEL MD database for NMA validation (https://mmb.irbbarcelona.org/MoDEL/); flexnMR NMR flexibility benchmark (verify URL).

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

ProDy (https://github.com/prody/ProDy) — Python NMA/ENM with GPU support via PyTorch; iModS server (https://imods.iqfr.csic.es) — NMA-based motion analysis; Bio3D R package (https://thegrantlab.org/bio3d/) — NMA in R; ElNemo (https://www.sciences.univ-nantes.fr/elnemo/) — elastic network modes server.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuSOLVER dense dsyevd for moderate-sized Hessians; cuSPARSE for sparse ANM matrix-vector products; custom CUDA Lanczos iteration for large sparse NMA; cuBLAS for B-factor RMSF accumulation. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
