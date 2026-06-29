# 2.3 — Cryo-EM Single-Particle Reconstruction

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🟢 Beginner · Established** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.3`
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

Single-particle cryo-EM reconstructs 3D density maps from thousands to millions of 2D projection images of vitrified protein particles in random orientations. The reconstruction pipeline involves CTF estimation, 2D class averaging, 3D ab initio reconstruction, and iterative 3D refinement (Bayesian polishing in RELION, non-uniform refinement in cryoSPARC). GPU acceleration is essential: the O(N·M) projection matching step (N particles × M reference projections) dominates walltime. RELION-3/4 and cryoSPARC achieve 10–100× GPU speedup over CPU. EMDB houses 50,000+ deposited maps.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Contrast transfer function (CTF) estimation, maximum a posteriori (MAP) 3D refinement, expectation-maximization (E-M) for orientation assignment, Fourier-Bessel reconstruction, Bayesian polishing, heterogeneous 3D classification, non-uniform refinement.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/cryo-em-single-particle-reconstruction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/cryo-em-single-particle-reconstruction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\cryo-em-single-particle-reconstruction.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: EMDB — 50,000+ cryo-EM density maps (https://www.ebi.ac.uk/emdb/); EMPIAR — raw cryo-EM particle images (https://www.ebi.ac.uk/empiar/); RCSB PDB structures with cryo-EM validation (https://www.rcsb.org); CryoDRGN benchmark datasets (https://github.com/ml-struct-bio/cryodrgn).

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

RELION (https://github.com/3dem/relion) — Bayesian cryo-EM reconstruction with CUDA GPU; cryoSPARC (https://cryosparc.com) — commercial GPU reconstruction platform; cryoDRGN (https://github.com/ml-struct-bio/cryodrgn) — heterogeneous VAE reconstruction with GPU; cisTEM (https://cistem.org) — GPU-accelerated cryo-EM software suite.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom CUDA kernels for Fourier-slice projection; cuFFT for 3D FFT reconstruction; GPU-batched 2D class averaging; warp-parallel expectation step; multi-GPU domain decomposition for large reconstructions. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
