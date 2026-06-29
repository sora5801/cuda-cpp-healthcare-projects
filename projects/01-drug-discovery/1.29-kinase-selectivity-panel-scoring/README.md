# 1.29 — Kinase Selectivity Panel Scoring

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.29`
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

Kinases share highly similar binding pockets, making selectivity a central challenge in kinase drug discovery. A GPU MD + ML pipeline can score a compound across 500+ kinase structures simultaneously: (1) GPU-parallel docking against all kinase homology models, (2) ML scoring using kinase-specific fingerprints (KLIFS features), (3) MM-GBSA rescoring. GPU acceleration allows a compound to be profiled against the entire kinome in minutes rather than days. Selectivity fingerprinting using ensemble docking with GPU makes this tractable.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Ensemble docking, kinase-ligand interaction fingerprints (KLIFS/IFP), selectivity scoring (SFP), homology model generation, structural kinome alignment, ML kinase activity prediction (KinaseML).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/kinase-selectivity-panel-scoring.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/kinase-selectivity-panel-scoring.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\kinase-selectivity-panel-scoring.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: KLIFS — kinase-ligand interaction fingerprinting database (https://klifs.net); KinomeScan — 468-kinase selectivity data (verify URL); ChEMBL kinase activity data (https://www.ebi.ac.uk/chembl/); DTC drug-target commons kinase panel (https://dtcommons.ai).

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

AutoDock-GPU (https://github.com/ccsb-scripps/AutoDock-GPU) — GPU docking against kinase panels; KLIFS Python API (https://github.com/volkamerlab/kissim) — kinase structural fingerprints; KinoML (https://github.com/openkinome/kinoml) — ML for kinase drug discovery; HTMD (https://github.com/Acellera/htmd) — GPU-based kinome docking workflows.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

GPU-parallel docking against kinase model array; GPU-batched IFP featurization; cuML for kinase activity ML training; Thrust for topK selectivity ranking. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
