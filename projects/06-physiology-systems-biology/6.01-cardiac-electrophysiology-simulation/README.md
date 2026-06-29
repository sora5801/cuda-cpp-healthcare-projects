# 6.1 — Cardiac Electrophysiology Simulation

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.1`
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

Simulates transmembrane voltage propagation across cardiac tissue by solving the monodomain or bidomain reaction-diffusion PDE coupled to stiff ODEs representing ionic channel kinetics (e.g., ten Tusscher-Panfilov, O'Hara-Rudy). Each voxel integrates 50–200 state variables per time step at sub-millisecond temporal resolution; a whole-heart simulation at 0.1 mm spatial resolution yields ~10⁸ nodes, making the per-node ODE update embarrassingly parallel. The GPU eliminates the otherwise serial per-cell Rush-Larsen / RL2 exponential gating integration. Operator splitting decouples the reaction (GPU-parallel ODE) from diffusion (sparse linear solve), and CUDA kernels saturate memory bandwidth on the former while cuSPARSE handles the latter.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Monodomain/bidomain reaction-diffusion, operator splitting (Strang/Godunov), Rush-Larsen explicit gating, Crank-Nicolson implicit diffusion, conjugate gradient with ILU(0) preconditioning, finite volume/finite element spatial discretization.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/cardiac-electrophysiology-simulation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/cardiac-electrophysiology-simulation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\cardiac-electrophysiology-simulation.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: PhysioNet MIT-BIH & MIMIC-III Waveform — 40 000+ ICU ECG/hemodynamic waveforms (https://physionet.org); CellML Physiome Repository — curated ionic cell models in CellML/SBML format importable by openCARP (https://models.physiomeproject.org); UK Biobank Cardiac MRI — 100 000+ cine CMR studies, access via application (https://www.ukbiobank.ac.uk); ACDC MICCAI Cardiac Challenge — 100-patient CMR with LV/RV/myocardium ground truth (https://www.creatis.insa-lyon.fr/Challenge/acdc/).

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

openCARP (https://git.opencarp.org/openCARP/openCARP) — MPI+CUDA cardiac EP solver, CARPutils Python scripting, v19.0 April 2026; MonoAlg3D_C (https://github.com/rsachetto/MonoAlg3D_C) — finite-volume GPU monodomain solver with Purkinje coupling and MPI batch dispatch; Cardioid/LLNL (https://github.com/llnl/cardioid) — multiscale cardiac suite (EP + mechanics + ECG), CUDA optional, Gordon Bell finalist; Chaste (https://github.com/Chaste/Chaste) — Oxford bidomain solver with cardiac mechanics module.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuSPARSE (diffusion SpMV), cuSOLVER (linear system), CUDA custom kernels (per-cell ODE Rush-Larsen); pattern: fine-grained thread-per-cell ODE + coarse SpMV for diffusion; streams for overlapping compute and halo exchange. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
