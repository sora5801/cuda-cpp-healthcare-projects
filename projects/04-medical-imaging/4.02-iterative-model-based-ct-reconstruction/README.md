# 4.2 — Iterative / Model-Based CT Reconstruction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.2`
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

Instead of a single analytical inversion, iterative methods repeatedly forward-project a current volume estimate, compare to measured sinogram data, then backproject the residual with statistical weighting. Penalized weighted least squares (PWLS) with total-variation (TV) or dictionary priors reduces noise by 30–50% at matched dose compared with FBP. Each outer iteration performs one full forward-projection and one backprojection — exactly the same GPU kernel bottleneck as FBP but repeated 20–200 times, making GPU mandatory for clinical throughput. ADMM decouples the data-fidelity and regularization sub-problems, enabling efficient GPU-friendly matrix-vector operations. Statistical models (Poisson likelihood for photon counts) can be incorporated for dose-optimal reconstruction.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

SIRT, SART, OS-EM for CT, PWLS-TV, PWLS with dictionary/wavelet priors, ADMM, primal-dual splitting (Chambolle-Pock), model-based iterative reconstruction (MBIR), plug-and-play ADMM with DnCNN denoiser.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/iterative-model-based-ct-reconstruction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/iterative-model-based-ct-reconstruction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\iterative-model-based-ct-reconstruction.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: 2016 AAPM Low-Dose CT Grand Challenge (https://www.aapm.org/grandchallenge/lowdosect/); Mayo Clinic Low-Dose CT dataset (available via TCIA); LIDC-IDRI via TCIA (https://www.cancerimagingarchive.net/).

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

ASTRA Toolbox (https://github.com/astra-toolbox/astra-toolbox) — GPU primitives, build iterative loops in Python/MATLAB; TIGRE (https://github.com/CERN/TIGRE) — includes OS-TV, SART, CGLS with GPU acceleration; ODL (Operator Discretization Library, https://github.com/odlgroup/odl) — Python framework wrapping ASTRA for variational reconstruction; LEAP (https://github.com/LLNL/LEAP) — LLNL GPU-accelerated CT reconstruction library with penalized-likelihood support.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuSPARSE (sparse system matrix), cuFFT, custom CUDA kernels for voxel-driven projection; outer loop on CPU, inner GPU kernel per OS subset; shared-memory tile reuse for cone-beam geometry. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
