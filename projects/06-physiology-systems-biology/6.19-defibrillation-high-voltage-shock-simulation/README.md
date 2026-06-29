# 6.19 — Defibrillation & High-Voltage Shock Simulation

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.19`
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

Defibrillation delivers a high-voltage electric field across the myocardium to terminate ventricular fibrillation. Simulating shock efficacy requires solving the bidomain equations driven by extracellular electrode currents, capturing virtual electrode polarization (VEP)—regions of depolarization and hyperpolarization induced at tissue boundaries—and subsequent re-entry termination. The nonlinear ionic response during shock (10 V/cm field, sub-ms timescale) and the fine spatial resolution needed (~0.1 mm) make GPU acceleration mandatory for whole-heart shock simulations.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Bidomain equations with extracellular stimulus, virtual electrode polarization theory, finite volume/element discretization, operator splitting with Rush-Larsen ionic integration, conjugate gradient linear solver, shock-protocol optimization (monophasic vs. biphasic), defibrillation threshold (DFT) estimation.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/defibrillation-high-voltage-shock-simulation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/defibrillation-high-voltage-shock-simulation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\defibrillation-high-voltage-shock-simulation.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: PhysioNet fibrillation/defibrillation recordings (https://physionet.org); openCARP defibrillation tutorial cases (https://opencarp.org); Cardioid (https://github.com/llnl/cardioid) — bidomain shock examples; patient-specific ICD placement datasets (verify institutional access).

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

openCARP (https://git.opencarp.org/openCARP/openCARP) — bidomain solver with extracellular stimulus for defibrillation studies; MonoAlg3D_C (https://github.com/rsachetto/MonoAlg3D_C) — GPU bidomain-capable extension; Cardioid/LLNL (https://github.com/llnl/cardioid) — cardiac EP + shock; Chaste (https://github.com/Chaste/Chaste) — bidomain with electrode boundary conditions.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuSPARSE conjugate gradient for bidomain elliptic solve; custom CUDA kernels for per-cell ionic ODE during shock timescale (0.01 ms dt); CUDA Unified Memory for large torso+heart mesh; pattern: dual-grid approach—fine heart mesh on GPU, coarse torso on CPU, coupled via interface boundary. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
