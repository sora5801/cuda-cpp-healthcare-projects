# 9.1 — Agent-Based Epidemic Simulation

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Epidemiology%20%26%20Public%20Health-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 9: Epidemiology & Public Health · Catalog ID `9.1`
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

Simulates individual-level epidemic spread across millions of synthetic agents, each with behavioural rules governing contact, infection, and recovery. GPU parallelism maps each agent to a thread or thread group: state updates (susceptible → exposed → infectious → recovered) are embarrassingly parallel across the population. The bottleneck is computing pairwise contacts within spatial proximity grids or synthetic social networks; cuGraph adjacency traversal accelerates this. Non-Markovian (renewal) dynamics require tracking each agent's infectious age distribution, a memory-intensive operation that fits within GPU SRAM when using compressed state representations. FlashSpread achieves end-to-end GPU execution with kernel-fused dense stepping.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

SIR/SEIR/SEIRD state machines per agent, contact kernel simulation (household, workplace, school stratification), non-Markovian renewal spreading, GPU-parallel BFS over contact graphs, Monte Carlo ensemble averaging, importance sampling for rare events, spatial hashing for local contact discovery.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/agent-based-epidemic-simulation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/agent-based-epidemic-simulation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\agent-based-epidemic-simulation.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: GLEAM / GLEaMviz global mobility + population data (https://www.gleamviz.org/) US Census TIGER/Line shapefiles + ACS commuting data (https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) Mossong et al. POLYMOD contact matrices — age-structured contact rates across 8 European countries (verify URL) SafeGraph / Dewey mobility data — retail foot traffic and mobility patterns (verify URL)

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

FRED (Framework for Reconstructing Epidemic Dynamics) (https://github.com/PublicHealthDynamicsLab/FRED) — individual-level US epidemic simulator FlashSpread (https://arxiv.org/abs/2604.22092) — end-to-end GPU framework for non-Markovian network spreading (verify GitHub URL) MEmilio (https://github.com/SciCompMod/memilio) — high-performance modular epidemic simulation software with GPU support Epiabm (https://github.com/RESIDE-ICL/epiabm) — GPU-parallelised ABM framework for epidemic simulation

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuGraph for contact network BFS/DFS, cuRAND for stochastic transition sampling, custom CUDA kernels for per-agent state update; pattern: one CUDA thread per agent with shared-memory contact lookup tables, warp-level primitives for neighbour enumeration. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
