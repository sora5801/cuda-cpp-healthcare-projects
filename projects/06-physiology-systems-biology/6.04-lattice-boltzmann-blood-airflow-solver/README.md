# 6.4 — Lattice-Boltzmann Blood/Airflow Solver

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.4`
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

The lattice-Boltzmann method (LBM) replaces continuum Navier-Stokes with a mesoscale kinetic equation for particle distribution functions on a regular grid—ideal for GPUs because each lattice site updates independently using only nearest-neighbor communication (the BGK collision step). Blood in complex vascular trees, red blood cell suspension rheology, and pulmonary airflow through bronchial trees all benefit from this approach. HemeLB achieves ~29.5 billion lattice site updates per second on thousands of cores; GPU versions (e.g., HemeLB GPU branch, PALABOS GPU) push throughput further with shared-memory streaming.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

BGK (Bhatnagar-Gross-Krook) collision operator, multi-relaxation time (MRT) LBM, D3Q19/D3Q27 velocity stencils, bounce-back boundary conditions for no-slip walls, Shan-Chen multiphase LBM, immersed boundary method for red blood cell membranes, Palabos fluid-particle coupling.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/lattice-boltzmann-blood-airflow-solver.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/lattice-boltzmann-blood-airflow-solver.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\lattice-boltzmann-blood-airflow-solver.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: PhysioNet coronary/aortic waveforms (https://physionet.org); Vascular Model Repository geometries (http://www.vascularmodel.com); open-access bronchial tree CT data from LIDC-IDRI (https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI); UK Biobank aortic flow MRI (https://www.ukbiobank.ac.uk).

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

HemeLB (https://github.com/hemelb-codes/hemelb) — sparse-geometry vascular LBM, MPI+GPU, scales to 32 000+ cores; HemePure GPU variant (https://github.com/hemelb-codes/HemePure) — cleaned GPU-first branch; PALABOS (https://gitlab.com/unigespc/palabos) — full-featured C++ LBM framework including multiphase and thermal extensions; USERMESO-2.0 (https://github.com/AnselGitAccount/USERMESO-2.0) — GPU red blood cell hemodynamics with deformable membrane.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom CUDA kernels for BGK streaming+collision in a single fused pass; shared memory for D3Q19 population arrays; texture memory for geometry masks; NCCL for GPU-direct halo exchange; pattern: one-thread-per-lattice-site with coalesced memory access on SOA layout. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
