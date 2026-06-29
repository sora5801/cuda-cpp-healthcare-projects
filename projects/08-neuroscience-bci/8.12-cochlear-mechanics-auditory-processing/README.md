# 8.12 — Cochlear Mechanics & Auditory Processing

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Neuroscience%20%26%20Brain--Computer%20Interfaces-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 8: Neuroscience & Brain-Computer Interfaces · Catalog ID `8.12`
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

The cochlea performs mechanical frequency decomposition via basilar membrane (BM) traveling waves, transforming sound to a tonotopic neural code via inner hair cells (IHCs) and auditory nerve fibers (ANFs). GPU simulation of a 3D BM model (finite element) or active cochlear model (outer hair cell electromotility — prestin) with coupled fluid mechanics and IHC/ANF spike generation supports hearing prosthesis design, audiogram prediction, and noise-induced hearing loss modeling.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

1D/2D/3D basilar membrane wave equation (FEM/FD), fluid-structure interaction for perilymph-BM coupling, outer hair cell electromotility (Prestin ODE), inner hair cell transducer (MET channel), auditory nerve fiber spike model (Zilany-Bruce), gammatone filterbank (frequency-domain equivalent), cochlear implant electrode models.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/cochlear-mechanics-auditory-processing.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/cochlear-mechanics-auditory-processing.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\cochlear-mechanics-auditory-processing.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: NH Hearing database (verify URL at nhlibrary.org); Auditory Model Toolbox benchmark datasets (https://amtoolbox.org); PhysioNet auditory brainstem response datasets (https://physionet.org); cochlear implant stimulation datasets from Cochlear Ltd (proprietary; verify institutional access).

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

CoNNear cochlea (https://github.com/HearingTechnology/CoNNear_cochlea) — PyTorch DNN cochlear mechanics model for real-time inference; mrkrd/cochlea (https://github.com/mrkrd/cochlea) — Python inner ear models interfacing NEURON/Brian; Auditory Model Toolbox (https://amtoolbox.org) — MATLAB/Octave/Python cochlear models; NEST simulator (https://github.com/nest/nest-simulator) — ANF population spiking.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuFFT for gammatone filterbank (bank of FIR filters via FFT convolution); custom CUDA FEM kernel for 1D BM wave equation (Thomas tridiagonal along BM length); batched ODE for ANF spike generation (one thread per fiber); pattern: frequency-band-parallel GPU computation, each warp handles one characteristic frequency band. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
