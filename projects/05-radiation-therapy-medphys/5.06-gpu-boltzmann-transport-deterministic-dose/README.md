# 5.6 — GPU Boltzmann Transport (Deterministic Dose)

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Radiation%20Therapy%20%26%20Medical%20Physics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 5: Radiation Therapy & Medical Physics · Catalog ID `5.6`
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

The linear Boltzmann transport equation (LBTE) describes radiation transport deterministically: it tracks the fluence distribution of particles as a function of position, direction, and energy without stochastic noise. Solving it on a clinical 6-DoF phase-space grid (x, y, z, θ, φ, E) discretized at clinical resolution yields a system with ~10⁹–10¹⁰ unknowns; iterative solvers (source iteration, diffusion synthetic acceleration) require GPU to be tractable. Acuros XB (Varian Eclipse) implements a GPU-accelerated LBTE solver that outperforms superposition-convolution in heterogeneous tissue. The 3D_RZ geometry and electron transport coupling make Boltzmann dose accurate in lung, bone/tissue interfaces where MC is preferred but slow.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Discrete ordinates (Sₙ) method, source iteration (SI), diffusion synthetic acceleration (DSA), multi-group energy discretization, linear discontinuous spatial FEM, Legendre polynomial scattering expansion, Acuros XB algorithm, coupled photon-electron LBTE.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/gpu-boltzmann-transport-deterministic-dose.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/gpu-boltzmann-transport-deterministic-dose.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\gpu-boltzmann-transport-deterministic-dose.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: AAPM TG-105 lung benchmark; IROC heterogeneity phantom datasets; IAEA photon cross-section library; Acuros XB validation datasets from Varian white papers (publicly documented).

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

OpenMC (https://github.com/openmc-dev/openmc) — open MC but with deterministic capabilities; Attila (commercial) and Denovo (https://github.com/ORNL-CEES/Exnihilo — verify URL) — deterministic transport; AHOTN (analytical and hybrid ordinates) codes (verify URL); GPU-accelerated Sₙ solvers in nuclear engineering literature (search "GPU Sn transport CUDA").

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuSPARSE for angular flux sweep (upwind differencing); cuFFT not applicable; custom CUDA kernel for inner transport sweep (spatial + angular decomposition); GPU memory: angular flux tensor in global memory, scattering source in shared memory; wavefront parallelism across spatial cells. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
