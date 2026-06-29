# 6.9 — Agent-Based Tissue / Immune Simulation

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.9`
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

Tissue is modeled as a population of autonomous agents (cells) each tracking position, velocity, cycle state, secretion rates, and mechanistic signaling. Cell-cell mechanical interactions (overlap repulsion, adhesion) require pairwise neighbor search that scales as O(N²) naively but drops to O(N) with spatial binning on GPU. Immune cell migration, cytokine diffusion, and tumor-immune coevolution are natural applications. PhysiCell supports 10⁵–10⁶ cells in 3D with GPU-accelerated substrate diffusion.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Center-based mechanics (soft-sphere repulsion + adhesion), cell cycle models (Ki67 basic/advanced, flow cytometry), substrate diffusion (Thomas ADI or explicit FD on Cartesian grid), chemotaxis gradient following, receptor-ligand binding kinetics, Boolean intracellular signaling (MaBoSS), spatial hashing for neighbor search.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/agent-based-tissue-immune-simulation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/agent-based-tissue-immune-simulation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\agent-based-tissue-immune-simulation.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: CancerSEA single-cell functional states (http://biocc.hrbmu.edu.cn/CancerSEA/); TCGA pan-cancer immune landscape (https://portal.gdc.cancer.gov); MIBI/IMC imaging mass cytometry datasets (various Zenodo deposits); TCIA immunotherapy imaging (https://www.cancerimagingarchive.net).

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

PhysiCell (https://github.com/MathCancer/PhysiCell) — 3D multicellular simulator with physics + biotransport; PhysiBoSS (https://github.com/PhysiBoSS/PhysiBoSS) — Boolean network–PhysiCell coupling for signaling; Chaste (https://github.com/Chaste/Chaste) — off-lattice cell-based models with vertex/Voronoi mechanics; MOOSE (https://github.com/BhallaLab/moose-core) — chemical signaling within cells.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom CUDA kernels for substrate PDE (explicit or ADI); CUDA Thrust for cell sort by spatial bin; atomic-add for cytokine source terms from agent loop; pattern: hybrid CPU (agent logic) + GPU (PDE + neighbor search) with pinned memory for cell state transfer. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
