# 8.15 — Optogenetics Stimulation Modeling

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Neuroscience%20%26%20Brain--Computer%20Interfaces-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 8: Neuroscience & Brain-Computer Interfaces · Catalog ID `8.15`
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

Optogenetics uses light-gated ion channels (channelrhodopsin-2, halorhodopsin) to activate or silence neurons with light. GPU simulation of an optogenetic stimulation experiment requires: (1) Monte Carlo photon transport in scattering brain tissue (thousands of photons per simulation, one independent random walk per photon → embarrassingly GPU-parallel); (2) ChR2 4-state photocycle ODE at each illuminated neuron; (3) network-level spiking response. Predicting light spread and activation volumes guides implant design.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Monte Carlo photon transport (random walk with scattering/absorption, Henyey-Greenstein phase function), ChR2 4-state kinetic model (Hegemann), 3-state simplified model, Beer-Lambert for superficial tissue, network SNN with light-activated conductance, activation map computation.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/optogenetics-stimulation-modeling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/optogenetics-stimulation-modeling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\optogenetics-stimulation-modeling.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Allen Brain Atlas gene expression for ChR2 targeting (https://portal.brain-map.org); DANDI optogenetics experimental datasets (https://dandiarchive.org); openMC photon transport validation cases (verify at openmc.org); OpenNeuro optogenetics fMRI datasets (https://openneuro.org).

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

MCX (Monte Carlo eXtreme) (https://github.com/fangq/mcx) — GPU-accelerated photon transport in biological tissue (CUDA, 1000× CPU speedup); GeNN (https://github.com/genn-team/genn) — SNN with ChR2 conductance kinetics; NEST simulator (https://github.com/nest/nest-simulator) — optogenetics module; NetPyNE (https://github.com/suny-downstate-medical-center/netpyne) — network simulation with optogenetic inputs.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom CUDA Monte Carlo kernel (one thread per photon packet, cuRAND for scattering events, atomic-add for fluence accumulation in voxel grid); cuSPARSE for network spike propagation; pattern: seed-parallel photon launch with cuRAND Sobol sequences, shared-memory partial fluence accumulation per thread-block. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
