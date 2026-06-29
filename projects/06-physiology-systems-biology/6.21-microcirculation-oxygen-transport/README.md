# 6.21 — Microcirculation & Oxygen Transport

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.21`
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

Oxygen delivery from red blood cells to tissue parenchyma involves convection in capillaries, diffusion through capillary walls and interstitium (Krogh cylinder / Green's function models), and intracellular O₂ reaction/consumption (Michaelis-Menten kinetics). A realistic tissue volume (~1 mm³) contains thousands of capillaries forming a 3D network; GPU parallelism is applied to the per-segment convection-diffusion solves and the volumetric Green's function integrals (which are an O(N²) operation accelerated to O(N log N) via multipole or GPU-NUFFT).

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Krogh cylinder O₂ transport, Green's function method (Secomb Hsu), 1D convection-diffusion along capillary segments, Michaelis-Menten O₂ consumption, fast multipole method (FMM) for Green's function sums, hemoglobin saturation curve (Hill equation), hematocrit-dependent RBC flux partitioning.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/microcirculation-oxygen-transport.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/microcirculation-oxygen-transport.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\microcirculation-oxygen-transport.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Vascular Model Repository (http://www.vascularmodel.com); two-photon microscopy microvascular datasets from Allen Institute (https://portal.brain-map.org); PhysioNet oxygen saturation waveforms (https://physionet.org); published microvascular network datasets (Secomb group, verify at secomb.org).

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

HemeLB (https://github.com/hemelb-codes/hemelb) — sparse LBM for capillary flow; USERMESO-2.0 (https://github.com/AnselGitAccount/USERMESO-2.0) — GPU red blood cell hemodynamics with deformable membranes; APBS (https://github.com/Electrostatics/apbs) — electrostatics solver repurposable for O₂ diffusion; OpenFOAM (https://github.com/OpenFOAM/OpenFOAM-dev) — volume-average tissue oxygenation.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CUDA NUFFT or FMM (cuFMM) for Green's function O₂ sums; custom CUDA kernels for per-segment RBC oxygen release; cuSPARSE for network flow solve; pattern: segment-parallel threads for convection update + shared-memory reduction for junction mass balance. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
