# 14.2 — Spatial / Whole-Cell Reaction-Diffusion at Molecular Resolution

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Emerging%2C%20Theoretical%20%26%20Grand--Challenge%20Frontiers-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 14: Emerging, Theoretical & Grand-Challenge Frontiers · Catalog ID `14.2`
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

Particle-based reaction-diffusion (PBRD) simulators track each molecule as an individual particle, enabling sub-micron spatial resolution of signaling gradients, receptor clustering, and organelle targeting. GPU-accelerated PBRD (Smoldyn GPU, ReaDDy GPU) parallelizes over molecules: each particle diffuses and reacts independently, with nearest-neighbor checks via GPU cell-list algorithms. A full cytoplasm simulation at molecular resolution for even a minimal cell (~500 K unique molecules) at physiologically relevant timescales (milliseconds) requires O(10¹²) timestep-particle updates — tractable only on multi-GPU systems. eGFRD (enhanced Green's Function Reaction Dynamics) is theoretically the most accurate but computationally costly, a prime GPU target.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Brownian dynamics with reaction (Smoluchowski), eGFRD Green's function propagators, interaction-site model (ISSA), diffusion-limited reaction kernel sampling, GPU cell-list O(N) neighbor search, reactive molecular dynamics.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/spatial-whole-cell-reaction-diffusion-at-molecular-resolution.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/spatial-whole-cell-reaction-diffusion-at-molecular-resolution.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\spatial-whole-cell-reaction-diffusion-at-molecular-resolution.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: CellOrganizer — generative models of subcellular morphology for simulation domains (http://www.cellorganizer.org/); PDB molecular crowding configurations; SBML-spatial format models (BioModels); MCell neural synapse models (https://mcell.org/).

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

ReaDDy (https://github.com/readdy/readdy) — GPU-accelerated particle-based RD (CPU + GPU backends); Smoldyn (https://github.com/ssandrews/Smoldyn) — off-lattice GPU-capable PBRD; MCell (https://mcell.org/) — Monte Carlo 3D reaction-diffusion for neurons; STEPS (https://github.com/CNS-OIST/STEPS) — tetrahedral-mesh spatial SSA with GPU support.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CUDA cell-list neighbor search (one thread per particle for neighbor pair collection), cuRAND for per-particle Brownian displacement sampling, Thrust for reaction-event sorting; pattern: GPU cell-list built from particle positions → parallel Brownian displacement → reaction probability check for each particle pair → acceptance-rejection sampling → time step advance. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
