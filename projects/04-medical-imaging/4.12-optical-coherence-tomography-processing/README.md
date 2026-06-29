# 4.12 — Optical Coherence Tomography Processing

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.12`
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

Spectral-domain OCT acquires spectra per A-scan (axial line); reconstruction requires dispersion compensation, interpolation from wavelength to wavenumber space, and 1D FFT per A-scan. A single B-scan of 2,048 A-scans × 2,048 spectral pixels requires 2,048 FFTs of length 2,048, easily parallelizable in GPU batches. Real-time 3D OCT volumes for surgical guidance require processing ~100 B-scans/second (~4 × 10⁸ FFT points/s), achievable only with GPU. Downstream retinal layer segmentation (8 boundaries, 3D graph search) and fluid detection (intra/subretinal, FLIO) add CNN inference workload; TensorRT-optimized U-Net achieves 3.5 ms/B-scan inference.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Spectral-domain FFT reconstruction, dispersion compensation, k-space resampling (NUFFT), GPU-batched FFT (cuFFT), 3D graph-cut layer segmentation, deep learning retinal layer segmentation (U-Net, U-NetRT), fluid detection CNN, Doppler OCT velocity mapping.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/optical-coherence-tomography-processing.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/optical-coherence-tomography-processing.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\optical-coherence-tomography-processing.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: OCTDL (https://www.nature.com/articles/s41597-024-03182-7) — 2,064 labeled OCT B-scans; Duke DME OCT dataset (https://people.duke.edu/~sf59/Chiu_BOE_2012_dataset.htm) — 110 annotated volumes; OCTA-500 (https://arxiv.org/abs/2012.07261) — OCT angiography volumes with labels.

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

OCT-Marker (https://github.com/neurodial/OCT-Marker) — annotation tool for OCT B-scans; Iowa Reference Algorithms (https://www.iibi.uiowa.edu/content/shared-software-Iowa-reference-algorithms) — graph-based segmentation (verify URL); k-Wave CUDA (https://github.com/klepo/k-Wave-Fluid-CUDA) — relevant for photoacoustic OCT extensions; real-time OCT reconstruction demos available in NVIDIA cuFFT samples.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuFFT batched 1D FFT (one FFT per A-scan, entire B-scan in one cuFFT call); custom CUDA kernel for dispersion phase correction; cuDNN for U-Net inference; CUDA streams for pipelining A-scan acquisition and reconstruction. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
