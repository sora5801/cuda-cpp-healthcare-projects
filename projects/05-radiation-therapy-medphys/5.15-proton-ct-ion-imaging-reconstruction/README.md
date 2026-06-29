# 5.15 — Proton CT & Ion Imaging Reconstruction

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Radiation%20Therapy%20%26%20Medical%20Physics-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 5: Radiation Therapy & Medical Physics · Catalog ID `5.15`
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

Proton CT (pCT) measures the residual range of individual protons after traversing a patient, converting to relative stopping power (RSP) maps directly for treatment planning — eliminating the Hounsfield-unit–to–RSP conversion uncertainty in X-ray CT. Each proton's path through tissue is a curved most-likely path (MLP) rather than a straight line; for 10⁸ protons per scan, computing all MLPs and binning them into a sinogram for reconstruction is a massively parallel GPU task. Iterative pCT reconstruction (POCS with RSP constraints, MLSD) requires forward/backprojection along curved proton paths, fundamentally different from X-ray cone-beam and requiring custom GPU geometry kernels. Clinical pCT scanners (IBA, PRaVDA) generate data at 10⁸ proton events/second — GPU is mandatory for any real-time capability.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Most-likely path (MLP) estimation (Highland formula, Gaussian scattering), list-mode proton CT reconstruction (CSPACS, MLSD), POCS with RSP box constraints, proton trajectory binning for FBP, iterative proton CT with scattering regularization, proton radiography for range verification.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/proton-ct-ion-imaging-reconstruction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/proton-ct-ion-imaging-reconstruction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\proton-ct-ion-imaging-reconstruction.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: PRaVDA proton CT datasets (verify URL); PRIMA proton CT consortium data (verify URL); TOPAS-generated pCT simulation data; ACE collaboration proton CT phantom datasets.

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

pCT reconstruction code from UCI/Santa Cruz collaboration (verify URL); TOPAS (https://github.com/OpenTOPAS/OpenTOPAS) — proton CT simulation; FRED (https://www.fredonline.eu/) — proton transport/range imaging; custom CUDA MLP projection repos (search GitHub "proton CT GPU most likely path").

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

One CUDA thread per detected proton (massively parallel MLP computation); cuBLAS for scattering covariance matrix updates; thrust sort for proton trajectory binning by projection angle; custom CUDA backprojection along curved MLP geometry; cuRAND for proton beam Monte Carlo sampling.

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
