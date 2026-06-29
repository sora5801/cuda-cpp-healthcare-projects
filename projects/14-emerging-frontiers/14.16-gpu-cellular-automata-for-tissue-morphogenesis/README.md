# 14.16 — GPU Cellular Automata for Tissue Morphogenesis

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Emerging%2C%20Theoretical%20%26%20Grand--Challenge%20Frontiers-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 14: Emerging, Theoretical & Grand-Challenge Frontiers · Catalog ID `14.16`
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

Lattice-Gas Cellular Automata (LGCA) and Cellular Automata (CA) models simulate tumor invasion, wound healing, and developmental tissue patterning at the cell scale on million-element grids. Every lattice site updates in parallel based on local neighborhood rules — a perfectly SIMT workload. GPU CA for tumor growth integrates nutrient diffusion (CUDA stencil), cell-cycle progression, and proliferation/death rules, enabling parameter sweeps over invasion phenotypes that would be intractable on CPU. Hybrid CA-PDE models couple discrete cell lattice (CUDA) with continuous nutrient/oxygen fields (CUDA finite difference).

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Lattice-Gas CA (LGCA) for cell migration, Cellular Automaton tumor model (Kansal-Torquato), Go-or-Grow phenotype switching, reaction-diffusion PDE for morphogens, Potts model for cell sorting, hybrid CA-FEM multiscale coupling.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/gpu-cellular-automata-for-tissue-morphogenesis.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/gpu-cellular-automata-for-tissue-morphogenesis.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\gpu-cellular-automata-for-tissue-morphogenesis.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: CancerOrganoid Drug Response Images (verify URL via Hubrecht); TCGA pathology slides for CA calibration (https://portal.gdc.cancer.gov/); CellMorph — time-lapse cell migration datasets (verify URL); Wound-Healing Assay Image Repository (verify URL via protocols.io).

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

PhysiCell (https://github.com/MathCancer/PhysiCell) — GPU-parallelized 3D agent-based tissue simulator; CompuCell3D (https://compucell3d.org/) — multi-algorithm tissue simulator with GPU support; CancerSim (https://github.com/joancalvente/cancersim) — GPU CA tumor growth code; Morpheus (https://morpheus.gitlab.io/) — spatial cell model simulation with GPU backend.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CUDA 2D/3D stencil kernels for CA lattice update, cuRAND for stochastic cell-fate decisions, Thrust for parallel phenotype census; pattern: N×N×N GPU lattice → one CUDA thread per lattice site → local rule evaluation → stochastic update → reaction-diffusion field update → time-step advance → GPU-rendered morphology export.

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
