# 4.3 — MRI Reconstruction with Compressed Sensing

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.3`
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

MRI acquires k-space (Fourier-domain) samples; compressed sensing (CS) reconstructs images from highly under-sampled k-space using sparsity priors (wavelet, total variation), enabling 4–8× scan acceleration. The core computation is a sequence of non-uniform FFTs (NUFFT/NFFT) for arbitrary k-space trajectories, followed by iterative soft-thresholding or proximal operators. NUFFT on a 3D grid at clinical resolution (~256³) involves ~10⁹ operations per iteration; GPU parallelism reduces each NUFFT to milliseconds vs. seconds on CPU, enabling real-time feedback. Multi-channel parallel imaging (SENSE, GRAPPA, PICS) adds per-coil FFTs (~32 channels), multiplying the compute by the coil count and making GPU essential.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

SENSE, GRAPPA, non-uniform FFT (NUFFT/NFFT3), PICS (Parallel Imaging + CS), Split-Bregman / ADMM, FISTA, total variation, wavelet sparsity, k-t SENSE for dynamic MRI.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/mri-reconstruction-with-compressed-sensing.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/mri-reconstruction-with-compressed-sensing.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\mri-reconstruction-with-compressed-sensing.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: fastMRI (NYU/Facebook, https://fastmri.med.nyu.edu/ and https://github.com/facebookresearch/fastMRI) — 1,500+ knee and 6,970+ brain raw k-space MRI scans; Calgary-Campinas-359 — multi-channel brain MRI k-space (https://sites.google.com/view/calgary-campinas-dataset/); SKM-TEA (Stanford knee MRI, https://github.com/StanfordMIMI/skm-tea).

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

BART (Berkeley Advanced Reconstruction Toolbox, https://github.com/mrirecon/bart) — production CS-MRI tool, GPU-accelerated PICS, SENSE, NUFFT; SigPy (https://github.com/mikgroup/sigpy) — Python GPU (CuPy) MRI signal-processing and NUFFT; MIRT (Michigan Image Reconstruction Toolbox, https://github.com/JeffFessler/MIRT.jl) — Julia/MATLAB iterative reconstruction with NUFFT; PyNUFFT (https://github.com/jyhmiinlin/pynufft) — Python NUFFT with CUDA/OpenCL backends.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuFFT for gridded FFT; custom CUDA NUFFT gridding kernels; cuBLAS for coil combination; per-coil FFT parallelized across CUDA streams; shared memory for gridding accumulation. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
