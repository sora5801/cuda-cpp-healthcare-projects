# 9.3 — Contact-Network & Graph Epidemic Dynamics

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Epidemiology%20%26%20Public%20Health-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 9: Epidemiology & Public Health · Catalog ID `9.3`
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

Simulates epidemic spread on empirical or synthetic contact networks where nodes are individuals and weighted edges encode contact intensity. GPU graph traversal (BFS/DFS) across networks with millions of nodes enables exploration of counterfactual intervention scenarios (edge removal, node vaccination) in seconds vs. hours on CPU. The Replay tool transforms empirical timestamped contact data into duration-weighted adjacency matrices and uses GPU sparse matrix operations for realistic epidemic simulation. cuGraph's PageRank and community detection accelerate identification of superspreader hubs for targeted interventions.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

SIR/SEIR stochastic simulation on contact graphs, Gillespie algorithm for continuous-time Markov chains, non-Markovian renewal kernels (FlashSpread), Belief Propagation for marginal inference on sparse graphs, community detection (Louvain, Leiden), targeted vaccination on high-degree nodes, R0 spectral radius estimation.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/contact-network-graph-epidemic-dynamics.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/contact-network-graph-epidemic-dynamics.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\contact-network-graph-epidemic-dynamics.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: SocioPatterns proximity contact data — face-to-face contacts in hospitals, schools, conferences (http://www.sociopatterns.org/) Copenhagen Networks Study — Bluetooth proximity + mobile data for 800 students (verify URL) GLEAM global mobility network (https://www.gleamviz.org/) NiemaGraphGen synthetic contact networks (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10038133/) — memory-efficient global-scale simulation toolkit

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

FlashSpread (https://arxiv.org/abs/2604.22092) — GPU framework for network epidemic dynamics (verify GitHub URL) Replay (https://link.springer.com/article/10.1186/s12911-025-03310-2) — GPU-accelerated temporal contact network epidemiology tool cuGraph (https://github.com/rapidsai/cugraph) — GPU graph analytics (PageRank, BFS, community detection) via RAPIDS EoN (Epidemics on Networks) (https://github.com/springer-math/Mathematics-of-Epidemics-on-Networks) — Python network epidemic simulation

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuGraph BFS/SSSP for infection spread on GPU-resident adjacency, cuSPARSE SpMV for transition probability matrices, cuRAND for stochastic edge activation; pattern: BFS-based wavefront parallelism with atomic state update per node. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
