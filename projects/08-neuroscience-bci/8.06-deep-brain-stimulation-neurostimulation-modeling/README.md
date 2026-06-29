# 8.6 — Deep Brain Stimulation / Neurostimulation Modeling

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Neuroscience%20%26%20Brain--Computer%20Interfaces-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 8: Neuroscience & Brain-Computer Interfaces · Catalog ID `8.6`
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

Deep brain stimulation (DBS) for Parkinson's disease delivers high-frequency (~130 Hz) electrical pulses from implanted electrodes in the subthalamic nucleus (STN). Predicting stimulation volume and network effects requires solving the quasi-static Poisson equation in a patient-specific brain volume conductor (DT-MRI-derived anisotropic conductivity), coupled to cable equation models of axons in the stimulation field. GPU parallelizes both the FEM Poisson solve and the hundreds of independent axon cable simulations needed to map activation thresholds.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Quasi-static Poisson equation (anisotropic conductivity from DTI), FEM on tetrahedral brain mesh, cable equation for myelinated axons (McNeal model, MRG model), chronaxie-rheobase threshold estimation, volume of tissue activated (VTA) mapping, network oscillation modeling (basal ganglia-thalamo-cortical loop ODEs).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/deep-brain-stimulation-neurostimulation-modeling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/deep-brain-stimulation-neurostimulation-modeling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\deep-brain-stimulation-neurostimulation-modeling.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: ADNI DT-MRI datasets (https://adni.loni.usc.edu); Human Connectome Project DT-MRI (https://db.humanconnectome.org); OpenNeuro DBS patient imaging (https://openneuro.org); OSS-DBS example cases (verify URL on github.com/OSS-DBSv2 or similar).

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

OSS-DBS v2 (https://github.com/SFB-ELAINE/OSS-DBS — verify URL) — open-source DBS simulation platform (FEM + axon models); SCIRun (https://github.com/SCIInstitute/SCIRun) — Utah electrodes + FEM neurostimulation; NetPyNE (https://github.com/suny-downstate-medical-center/netpyne) — basal ganglia network models for DBS effect simulation; NEURON (https://github.com/neuronsimulator/nrn) — canonical axon cable equation solver.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuSPARSE CG for anisotropic FEM Poisson solve; batch cable ODE across hundreds of axon trajectories (cuSOLVER batched tridiagonal or custom Thomas algorithm CUDA kernel); cuBLAS for DBS-induced voltage interpolation; pattern: parallel FEM solve then embarrassingly parallel axon threshold sweeps. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
