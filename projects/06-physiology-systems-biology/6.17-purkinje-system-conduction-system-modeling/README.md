# 6.17 — Purkinje System & Conduction System Modeling

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.17`
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

The cardiac conduction system (sinoatrial node, AV node, His bundle, bundle branches, Purkinje fiber network) initiates and coordinates ventricular activation. Simulating the Purkinje tree requires a 1D cable equation solver on a fractal branching network of ~10⁵ segments, coupled at Purkinje-muscle junctions (PMJs) to the 3D ventricular myocardium. GPU parallelism across the large number of independent 1D cable segments accelerates conduction pathway simulations for pacemaker dysfunction and re-entry arrhythmia studies.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

1D cable equation (monodomain) on Purkinje tree, PMJ coupling via gap-junction conductance, Stewart-Zhang Purkinje ionic model, His-Purkinje conduction velocity calibration, tree generation algorithms (L-system or rule-based branching), graph-based conduction delay computation.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/purkinje-system-conduction-system-modeling.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/purkinje-system-conduction-system-modeling.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\purkinje-system-conduction-system-modeling.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: openCARP community Purkinje experiments (https://opencarp.org/community/community-experiments); MonoAlg3D_C Purkinje examples (https://github.com/rsachetto/MonoAlg3D_C); NeuroMorpho (morphological analogy for tree datasets) (https://neuromorpho.org); PhysioNet His-bundle electrogram databases (https://physionet.org).

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

MonoAlg3D_C (https://github.com/rsachetto/MonoAlg3D_C) — GPU monodomain solver with integrated Purkinje network and PMJ calibration; openCARP (https://git.opencarp.org/openCARP/openCARP) — supports Purkinje cable coupling; Cardioid/LLNL (https://github.com/llnl/cardioid) — includes Purkinje conduction modeling; Chaste (https://github.com/Chaste/Chaste) — 1D cable equation infrastructure.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Batch tridiagonal solvers (cuSPARSE batched Thomas) for 1D cable segments; custom CUDA kernels for ionic ODEs at each Purkinje node; CUDA graph for recurring per-beat computation pattern; pattern: one thread per Purkinje node, shared memory for tridiagonal coefficients within a segment. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
