# 10.12 — Microfluidic Device & Organ-on-Chip Simulation

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Biomechanics%2C%20Biomedical%20Devices%20%26%20Surgery-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 10: Biomechanics, Biomedical Devices & Surgery · Catalog ID `10.12`
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

Lab-on-a-chip and organ-on-chip devices feature micrometer-scale channels where Re < 1 and Péclet numbers span orders of magnitude, demanding accurate Navier-Stokes + advection-diffusion solutions on geometrically complex domains. Lattice-Boltzmann Method (LBM) maps perfectly to GPU: each lattice node streams and collides independently, achieving memory-bandwidth-bound performance near GPU peak. GPU LBM-DEM (discrete element method) co-simulates cell transport, adhesion, and deformation through microchannels. Design optimization of pillar geometry, channel bifurcations, and gradient generators runs via adjoint sensitivity on GPU, drastically accelerating the design-of-experiment cycle for organ-chip platforms.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

D3Q19/D3Q27 LBM with BGK or MRT collision, immersed boundary coupling for deformable cells, lattice-DEM for rigid particle transport, advection-diffusion for chemical gradient generation, adjoint sensitivity analysis for geometry optimization.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/microfluidic-device-organ-on-chip-simulation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/microfluidic-device-organ-on-chip-simulation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\microfluidic-device-organ-on-chip-simulation.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Microfluidic Gradient Generator Benchmark (LBM validation, Zenodo); PhysioMimetics organ-chip flow data (verify URL); OpenFOAM microfluidic validation cases (https://www.openfoam.com/); Glioblastoma-on-chip CFD dataset (Frontiers Bioeng 2025) (verify Zenodo).

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

Palabos (https://gitlab.com/unigespc/palabos) — GPU-capable LBM library for complex fluid dynamics; LEDDS (https://arxiv.org/abs/2512.04997) — portable LBM-DEM GPU simulations; waLBerla (https://www.walberla.net/) — massively parallel LBM framework with GPU support; OpenFOAM (https://github.com/OpenFOAM) — with GPU-accelerated linear solvers via PETSc-CUDA backend.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CUDA kernels for per-node stream-and-collide (one thread per lattice node), cuFFT for spectral pressure solve, Thrust for particle tracking; pattern: GPU-resident 3D lattice → CUDA stream-and-collide kernel → IBM force spreading for deformable cells → chemical concentration advection-diffusion update → device geometry optimization via adjoint. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
