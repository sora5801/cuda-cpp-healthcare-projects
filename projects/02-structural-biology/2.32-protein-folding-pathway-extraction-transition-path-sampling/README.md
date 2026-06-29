# 2.32 — Protein Folding Pathway Extraction (Transition Path Sampling)

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Structural%20Biology%20%26%20Protein%20Science-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 2: Structural Biology & Protein Science · Catalog ID `2.32`
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

Transition Path Sampling (TPS) harvests rare folding/unfolding events by shooting from configurations near the transition state and accepting/rejecting trajectories that connect folded and unfolded basins. GPU MD makes it practical to run many short (~1–100 ns) shooting moves in parallel. AIMMD (AI-augmented MD) uses GPU-trained neural networks to identify committor isosurfaces, accelerating TPS convergence. Applications include protein folding mechanism elucidation, cryptic pocket opening pathways, and drug unbinding kinetics (τRAMD, WExplore).

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Transition path sampling (TPS) shooting move, aimless shooting, committor analysis, path collective variables (PathCV), weighted ensemble sampling (WExplore/WE-H), τRAMD unbinding kinetics, AIMMD neural committor.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/protein-folding-pathway-extraction-transition-path-sampling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/protein-folding-pathway-extraction-transition-path-sampling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\protein-folding-pathway-extraction-transition-path-sampling.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Anton/Shaw millisecond trajectories as TPS starting configurations; GPCRmd pathway datasets (https://gpcrmd.org); folding benchmarks: Trp-cage, chignolin, WW domain; SAMPL host-guest kinetics challenges (verify URL).

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

OpenPathSampling (https://github.com/openpathsampling/openpathsampling) — TPS on GPU via OpenMM; HTMD (https://github.com/Acellera/htmd) — GPU-accelerated adaptive sampling; WESTPA (https://westpa.github.io/westpa/) — weighted ensemble sampling on GPU MD; AIMMD (https://github.com/bioRxiv AIMMD, verify URL) — AI-augmented TPS with GPU neural committor.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

GPU MD for fast shooting trajectories; GPU neural network committor inference in AIMMD; NCCL for WE parent-child trajectory coordination; embarrassingly parallel independent shooter array on multi-GPU. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
