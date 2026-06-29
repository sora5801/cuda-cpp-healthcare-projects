# 6.14 — Multi-Scale Physiological Modeling

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.14`
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

Couples models operating at different spatial/temporal scales: molecular (ion channel kinetics, μs–ms), cellular (action potential, ms), tissue (wave propagation, ms–s), organ (cardiac output, heartbeat), and system (circulation, minutes). The computational challenge is that fine-scale models (cell ODE) must be solved at each quadrature point of a coarse FEM mesh simultaneously—yielding millions of ODE instances per time step. GPU batch-ODE solving (CVODE GPU) fills this role. The Virtual Physiological Human (VPH) framework coordinates inter-scale coupling.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Heterogeneous multiscale method (HMM), operator splitting for scale coupling, homogenization, batch CVODE for cell-level ODEs at FEM quadrature points, Windkessel/1D vessel network for circulation, FEM for organ-level mechanics/EP, co-simulation coupling (FMI standard).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/multi-scale-physiological-modeling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/multi-scale-physiological-modeling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\multi-scale-physiological-modeling.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Physiome Model Repository — VPH-standard CellML models (https://models.physiomeproject.org); BioModels Database (https://www.ebi.ac.uk/biomodels); UK Biobank multi-modal phenotyping (https://www.ukbiobank.ac.uk); OpenCMISS examples (https://github.com/OpenCMISS/examples).

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

OpenCMISS/cm (https://github.com/OpenCMISS/cm) — multi-physics multi-scale FEM framework; SUNDIALS batch CVODE GPU (https://github.com/LLNL/sundials) — batch ODE for sub-grid cell models; simcardems (https://github.com/ComputationalPhysiology/simcardems) — cardiac electromechanics multi-scale coupling; Chaste (https://github.com/Chaste/Chaste) — multi-scale cardiac + lung + tumor modeling.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

SUNDIALS CUDA NVector + batch CVODE (cell ODE at quadrature points); cuSPARSE for coarse-mesh FEM assembly; CUDA streams for asynchronous scale coupling; pattern: two-level parallelism—CUDA grid over FEM elements, threads over per-element ODE RHS evaluation. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
