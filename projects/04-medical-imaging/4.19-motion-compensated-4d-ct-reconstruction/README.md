# 4.19 — Motion-Compensated 4D-CT Reconstruction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.19`
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

4D-CT captures respiratory motion by sorting ~4,000 projections into 10 breathing phases, then reconstructing each phase — effectively 10 independent 3D reconstruction problems with very few (~400) projections each (severe under-sampling). Simultaneous motion-compensated reconstruction (MCR) jointly estimates the reference volume and DVF by alternating between image reconstruction and non-rigid registration steps, each of which is a GPU-intensive computation. 4D-CBCT for adaptive radiotherapy is even more challenging (sparser projections, imaging dose constraints) and requires GPU-accelerated iterative reconstruction with motion-model regularization. Deep learning methods (4D Gaussian splatting, score-based priors) now push 4D-CBCT quality toward 4D-CT standards using GPU-trained priors.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Phase-binning and amplitude-binning 4D sorting, McKinnon-Bates 4D FDK, simultaneous MCR (PICCS, ROOSTER), GPU SART with deformable motion model, respiratory motion model (PCA-based surrogate), 4D neural radiance fields, 4D Gaussian splatting reconstruction.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/motion-compensated-4d-ct-reconstruction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/motion-compensated-4d-ct-reconstruction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\motion-compensated-4d-ct-reconstruction.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: DIR-Lab 4D-CT lung dataset (https://www.dir-lab.com/) — 10 cases with expert landmark pairs; TCIA 4D-CT lung radiotherapy collections; POPI model (https://www.creatis.insa-lyon.fr/rio/popi-model); CIRS dynamic lung phantom data.

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

RTK (https://github.com/RTKConsortium/RTK) — 4D ROOSTER motion-compensated reconstruction; ASTRA (https://github.com/astra-toolbox/astra-toolbox) — GPU projection kernels for 4D iterative; TIGRE (https://github.com/CERN/TIGRE) — 4D-capable iterative; Plastimatch (https://plastimatch.org/) — DIR integration with 4D dose.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

GPU SART kernel for each phase subset; CUDA Demons for inter-phase registration; cuFFT for motion model PCA basis; texture memory for 4D DVF interpolation; alternating GPU compute between reconstruction and registration steps. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
