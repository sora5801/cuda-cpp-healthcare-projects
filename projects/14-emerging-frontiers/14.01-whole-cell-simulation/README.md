# 14.1 — Whole-Cell Simulation

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Emerging%2C%20Theoretical%20%26%20Grand--Challenge%20Frontiers-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 14: Emerging, Theoretical & Grand-Challenge Frontiers · Catalog ID `14.1`
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

Whole-cell simulation aspires to mechanistically model every gene, mRNA, protein, metabolite, and organelle in a single bacterium or yeast cell simultaneously. The scale challenge is staggering: E. coli has ~4,300 genes and ~1.5 M ribosomes; a complete stochastic reaction-diffusion simulation at molecular resolution would require centuries on a single CPU. GPU acceleration of spatial SSA (Gillespie/tau-leaping) over a discretized cell volume enables partial whole-cell models (gene expression + metabolism) to run in tractable time. The STEPS simulator (parallel, GPU-accelerated) handles reaction-diffusion on tetrahedral meshes representing subcellular geometry. Achieving true whole-cell simulation likely requires exascale GPU clusters.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Spatial Gillespie SSA (Next Subvolume Method / ISSA), tau-leaping with error control, next-reaction method (NRM), multiscale hybrid: ODE for deterministic fast species + SSA for rare events, GPU-parallel lattice microbes (LM) algorithm, whole-cell model composition (FBA + transcription/translation + signaling).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/whole-cell-simulation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/whole-cell-simulation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\whole-cell-simulation.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Mycoplasma genitalium whole-cell model (Karr et al. Cell 2012) parameters (https://simtk.org/projects/wc_models); E. coli K-12 transcriptomics (GEO GSE2198 and related); BioModels Database whole-cell models (https://www.ebi.ac.uk/biomodels/); JCVI Syn3A minimal genome datasets (https://www.jcvi.org/research/first-minimal-synthetic-bacterial-cell).

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

STEPS (https://github.com/CNS-OIST/STEPS) — GPU-accelerated stochastic spatial reaction-diffusion in tetrahedral meshes; Lattice Microbes (LM) (https://github.com/Luthey-Schulten-Lab/Lattice_Microbes) — GPU spatial stochastic simulator for E. coli; Smoldyn (https://github.com/ssandrews/Smoldyn) — off-lattice particle-based RD simulator (multi-GPU); WholeCellKB (https://github.com/CovertLab/WholeCell) — Karr whole-cell model framework.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CUDA kernels for parallel subvolume SSA reaction firing, cuRAND for per-subvolume random streams, NCCL for multi-GPU spatial domain decomposition; pattern: cell volume partitioned into tetrahedral subvolumes on GPU → parallel SSA firing per subvolume → diffusive transfer between subvolumes via CUDA inter-thread communication → global species count aggregation → repeat at nanosecond timescale. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
