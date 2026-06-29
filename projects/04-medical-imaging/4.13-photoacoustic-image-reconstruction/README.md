# 4.13 — Photoacoustic Image Reconstruction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.13`
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

Photoacoustic imaging (PAI) generates ultrasound waves by pulsed laser absorption in tissue; images are reconstructed from time-series pressure data on a sensor surface. Delay-and-sum backprojection is analogous to ultrasound but in 3D; for 1,024 sensors and a 256³ volume, ~68 billion delay-and-sum operations are required per image — tractable only on GPU. Model-based iterative reconstruction solves the wave equation numerically (k-space pseudospectral method via cuFFT), enabling quantitative PAI with accurate acoustic attenuation and heterogeneous speed-of-sound modelling. Real-time 3D PA imaging for interventional guidance requires GPU throughput of multiple frames/second.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Delay-and-sum backprojection, time-reversal reconstruction, universal back-projection, k-space pseudo-spectral wave propagation (k-Wave), iterative model-based PA reconstruction, compressed sensing PAI, deep learning end-to-end PA reconstruction.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/photoacoustic-image-reconstruction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/photoacoustic-image-reconstruction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\photoacoustic-image-reconstruction.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: k-Wave simulation datasets (generated locally); USCT (Ultrasound Computed Tomography) benchmark data (verify URL); in vivo photoacoustic datasets from Nature Communications publications (open access); PASCAA challenge datasets (verify URL at photoacoustics.org).

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

k-Wave (http://www.k-wave.org/, CUDA C++ version at https://github.com/klepo/k-Wave-Fluid-CUDA) — industry-standard PA/US simulation and reconstruction toolbox; OpenMSOT (open multi-spectral optoacoustic tomography framework, verify URL); k-Wave MATLAB + CUDA backend for fast GPU wave simulation; PyTomography (https://github.com/lukepolson/pytomography) — Python GPU tomographic reconstruction including photoacoustic.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuFFT for k-space wave propagation; custom CUDA kernel for DAS (one thread per voxel, loop over sensors); CUDA texture for time-series data interpolation; shared memory for sensor geometry LUT; multi-GPU decomposition over k-space planes. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
