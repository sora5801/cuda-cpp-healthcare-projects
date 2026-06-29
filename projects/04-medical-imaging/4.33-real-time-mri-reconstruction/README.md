# 4.33 — Real-Time MRI Reconstruction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.33`
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

Interventional and cardiac MRI require image reconstruction latency <100 ms to enable real-time guidance (catheter navigation, cardiac function monitoring). Online adaptive compressed sensing with sliding window or XD-GRASP (extra-dimensional GRASP) processes continuously acquired non-Cartesian k-space (radial, spiral) with GPU NUFFT and compressed sensing reconstruction running in a locked pipeline with acquisition. Gadgetron, an open-source streaming MR reconstruction framework, pipelines coil compression, NUFFT, GRAPPA, and deep learning inference on GPU with acquisition-synchronous operation. The cardiac cycle adds a gating dimension, requiring 4D (3D + cardiac phase) reconstruction at interactive speeds only feasible on GPU.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

XD-GRASP (multi-dimensional golden-angle radial), sliding-window NUFFT, online GRAPPA, low-rank + sparse reconstruction, compressed sensing NUFFT with TV, cardiac-gated CS (XTREAM, L+S), neural network real-time reconstruction (MoDL-S), real-time MRI with physiological monitoring.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/real-time-mri-reconstruction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/real-time-mri-reconstruction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\real-time-mri-reconstruction.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Cardiac MRI datasets from ACDC challenge (https://www.creatis.insa-lyon.fr/Challenge/acdc/); CMRxRecon 2023 challenge (https://cmrxrecon.github.io/); dynamic cardiac MRI from MRXCAT simulation (verify URL); real-time fetal MRI from research groups.

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

Gadgetron (https://github.com/gadgetron/gadgetron) — GPU streaming MRI reconstruction server, GRAPPA/NUFFT/DL plugins; BART (https://github.com/mrirecon/bart) — GPU GRASP/CS-MRI for batch; MRzero (https://github.com/MRsimulator/MRzero) — differentiable real-time MR simulation; SigPy (https://github.com/mikgroup/sigpy) — Python NUFFT/CUDA for real-time prototyping.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuFFT for NUFFT gridding; CUDA streams for acquisition-synchronous pipeline (double-buffering: acquire on CPU/scanner, reconstruct on GPU simultaneously); cuDNN for online DL inference; CUDA thrust for dynamic radial k-space sorting; multi-GPU for parallel cardiac phase reconstruction. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
