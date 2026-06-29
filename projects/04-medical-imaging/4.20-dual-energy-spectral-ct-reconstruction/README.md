# 4.20 — Dual-Energy / Spectral CT Reconstruction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.20`
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

Dual-energy CT (DECT) acquires sinograms at two X-ray spectra (e.g., 80 kV and 140 kV) to enable material decomposition (separating water vs. iodine basis materials, or bone vs. soft tissue). Material decomposition in projection space requires solving a 2×2 nonlinear system per sinogram bin (~10⁸ bins), each requiring Newton iteration — trivially parallel across bins on GPU. Photon-counting CT (PCCT) extends this to 4–8 energy bins, increasing the system size to 8×8 and multiplying GPU compute by 4× but enabling K-edge imaging of contrast agents. Image-domain decomposition avoids projection-space issues but requires iterative reconstruction at each energy.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Projection-domain material decomposition (Newton iteration per sinogram bin), image-domain material decomposition, basis-material iterative CT (ADMM), virtual monoenergetic imaging, K-edge subtraction, photon-counting spectral reconstruction, GPU splitting-based DECT ADMM.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/dual-energy-spectral-ct-reconstruction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/dual-energy-spectral-ct-reconstruction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\dual-energy-spectral-ct-reconstruction.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: AAPM Spectral CT challenge datasets (verify URL at aapm.org); MARS photon-counting CT datasets (https://www.marsbioimaging.com/); TCIA DECT collections; simulated DECT from published XCAT phantom.

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

ASTRA (https://github.com/astra-toolbox/astra-toolbox) — multi-energy projection/backprojection primitives; TIGRE (https://github.com/CERN/TIGRE) — spectral CT reconstruction; ODL (https://github.com/odlgroup/odl) — material decomposition operators; splitting-based GPU DECT paper code (https://arxiv.org/abs/1905.00934 — verify repo link in paper).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom CUDA kernel for per-bin Newton iteration (one thread per sinogram bin, 2×2 system solve in registers); cuFFT for spectral filter; shared memory for energy-bin grouped bins; cuBLAS for joint iterative reconstruction across energy channels. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
