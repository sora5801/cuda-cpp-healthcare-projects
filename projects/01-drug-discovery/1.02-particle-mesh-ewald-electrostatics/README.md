# 1.2 — Particle-Mesh Ewald Electrostatics

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟢 Beginner · Established** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.2`
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

Long-range electrostatics in periodic MD systems cannot be truncated without severe artifacts; PME splits the Coulomb sum into a short-range real-space part (evaluated with cutoff) and a smooth long-range reciprocal-space part evaluated on a 3D grid via FFT. The GPU acceleration opportunity is two-fold: the charge spreading (particle-to-mesh) and force interpolation (mesh-to-particle) steps are data-parallel over atoms, while the 3D FFT is handled by cuFFT. PME scales as O(N log N) and dominates walltime for large biological systems. Achieving double-precision accuracy at float throughput is the main engineering challenge.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Ewald summation, B-spline charge interpolation (order 4–6), 3D FFT on GPU, real-space erfc damping, smooth PME (SPME), Particle-Particle Particle-Mesh (P3M).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/particle-mesh-ewald-electrostatics.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/particle-mesh-ewald-electrostatics.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\particle-mesh-ewald-electrostatics.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: CHARMM-GUI solvation benchmark sets — pre-built periodic protein-water boxes (https://charmm-gui.org); D. E. Shaw Research Anton trajectories — ms-scale trajectory archives (available via DE Shaw); ion channel benchmark systems (MemProtMD, https://memprotmd.bioch.ox.ac.uk).

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

GROMACS CUDA PME (https://github.com/gromacs/gromacs) — reference GPU PME implementation; NAMD GPU PME (https://www.ks.uiuc.edu/Research/namd/) — tiled domain-decomposed PME; OpenMM PME plugin (https://github.com/openmm/openmm) — Python-accessible PME with mixed-precision; cuFFT (https://developer.nvidia.com/cufft) — NVIDIA's FFT library used internally by all above.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuFFT for 3D FFT; custom CUDA kernels for B-spline charge spreading (atom-parallel) and gradient interpolation; shared-memory tiling to minimize global memory traffic; atomics for scatter-add accumulation on the charge grid. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
