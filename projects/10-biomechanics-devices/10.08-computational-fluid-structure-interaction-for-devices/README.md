# 10.8 — Computational Fluid-Structure Interaction for Devices

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Biomechanics%2C%20Biomedical%20Devices%20%26%20Surgery-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 10: Biomechanics, Biomedical Devices & Surgery · Catalog ID `10.8`
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

Heart valves, stents, LVADs, and arterial stents involve tightly coupled incompressible fluid (blood) and elastic/rigid solid (leaflets, walls) dynamics that must be co-solved. Immersed boundary methods (IBM) embed flexible structures in Eulerian fluid grids, requiring interpolation and spreading operations that are GPU-parallelized across boundary points. SPH (smoothed particle hydrodynamics) replaces grids with Lagrangian particles, enabling free-surface and high-deformation flows suitable for LVAD impeller modeling. The FSEI-GPU code solves fluid-structure-electrophysiology interaction of the full left heart on a few GPU cards, completing one heartbeat in hours instead of days. Multi-GPU domain decomposition via NCCL enables scaling to whole-cardiovascular-system models.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Immersed Boundary Method (IBM), Lattice-Boltzmann Method (LBM), ISPH/TLSPH Smoothed Particle Hydrodynamics, arbitrary Lagrangian-Eulerian (ALE) formulation, Navier-Stokes fractional-step solver, hemolysis (GKM model) and thrombosis (biochemical agonist) submodels.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/computational-fluid-structure-interaction-for-devices.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/computational-fluid-structure-interaction-for-devices.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\computational-fluid-structure-interaction-for-devices.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: 4D Flow MRI Benchmark (HEArt) — time-resolved 3D velocity fields in cardiac chambers (https://arxiv.org/abs/2111.00720); HeartFlow FFRCT coronary dataset (commercial, academic access); Aortic Flow Simulation Database from SimVascular (https://simvascular.github.io/); OpenHeart MRI cohort — segmented cardiac geometries (verify URL via Zenodo).

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

FSEI-GPU (https://arxiv.org/abs/2103.15187) — CUDA Fortran FSI+electrophysiology heart solver (see ScienceDirect for code link); SimVascular (https://github.com/SimVascular/SimVascular) — patient-specific cardiovascular FSI pipeline; GPU-accelerated IB solver (Bhalla group, https://arxiv.org/html/2605.04335) — OpenACC + CUDA + NCCL extreme-scale IBM; PyFR (https://github.com/PyFR/PyFR) — GPU-native high-order Navier-Stokes solver adaptable to biofluid domains.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CUDA kernels for IBM force-spreading/interpolation, cuFFT for Poisson pressure solve, NCCL for multi-GPU halo exchange, cuSPARSE for FSI coupling matrix; pattern: Eulerian fluid grid partitioned across GPUs → IBM Lagrangian marker forces spread to fluid grid via CUDA kernel → pressure solve via FFT → structure positions updated → halo exchange via NCCL. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
