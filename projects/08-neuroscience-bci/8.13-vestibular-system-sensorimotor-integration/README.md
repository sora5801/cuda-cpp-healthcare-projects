# 8.13 — Vestibular System & Sensorimotor Integration

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Neuroscience%20%26%20Brain--Computer%20Interfaces-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 8: Neuroscience & Brain-Computer Interfaces · Catalog ID `8.13`
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

The vestibular system detects head motion via semicircular canals (angular velocity → cupula deflection → hair cell activation) and otolith organs (linear acceleration). GPU simulation of the full cupula-endolymph fluid-structure interaction (FSI) in all three canals plus otolith membrane mechanics, coupled to downstream neural coding (irregular vs. regular afferents) and central vestibulo-ocular reflex (VOR) circuitry, is computationally demanding but tractable with GPU. Applications include space medicine, motion sickness modeling, and vestibular implant design.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Cupula-endolymph FSI (Stokes flow + elastic membrane), hair bundle adaptation ODE, afferent spike coding (van Hemmen model), torsion pendulum model, Kalman-filter Bayesian internal model, VOR motor command ODE, cerebellar Purkinje cell learning (Marr-Albus-Ito).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/vestibular-system-sensorimotor-integration.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/vestibular-system-sensorimotor-integration.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\vestibular-system-sensorimotor-integration.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Vestibular electrophysiology data from DANDI (https://dandiarchive.org); Human Connectome Project functional connectivity (vestibular cortex) (https://db.humanconnectome.org); PhysioNet balance/posturography datasets (https://physionet.org); published cupula FSI experimental datasets (verify via institutional access).

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

NEST simulator (https://github.com/nest/nest-simulator) — vestibular afferent and VOR circuit models; GeNN (https://github.com/genn-team/genn) — GPU SNN for VOR + cerebellar learning; OpenFOAM (https://github.com/OpenFOAM/OpenFOAM-dev) — semicircular canal endolymph FSI; FEBio (https://github.com/febiosoftware/FEBio) — otolith membrane FEM.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom CUDA Stokes flow solver for endolymph; batch ODE for hair bundle + afferent dynamics (one thread per hair cell); cuBLAS for cerebellar parallel fiber weight matrix updates; pattern: fluid-structure coupling via immersed boundary method on GPU with split-step FSI. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
