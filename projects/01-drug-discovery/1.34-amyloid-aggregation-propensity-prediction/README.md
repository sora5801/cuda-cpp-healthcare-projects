# 1.34 — Amyloid / Aggregation Propensity Prediction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.34`
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

Protein aggregation drives diseases (Alzheimer's, Parkinson's, ALS) and is a major liability in biologic drug development. GPU-accelerated coarse-grained and atomistic MD can directly simulate fibril nucleation and extension, but requires microsecond-to-millisecond timescales accessible only with GPU enhanced sampling. ML aggregation predictors (AGGRESCAN3D, CamSol) train on experimental aggregation rates; GPU-trained GNNs on protein sequence+structure outperform sequence-only models. Amyloid fibril cryo-EM structures from EMDB drive validation.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

β-aggregation propensity scoring, coarse-grained MD of oligomerization (MARTINI), REMD/MetaD of early aggregation, GNN aggregation predictor, solubility prediction neural networks.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/amyloid-aggregation-propensity-prediction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/amyloid-aggregation-propensity-prediction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\amyloid-aggregation-propensity-prediction.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: AmyPro — curated amyloidogenic sequence database (https://amypro.net); FoldAmyloid prediction database (verify URL); ThT fluorescence assay aggregation kinetics datasets; EMDB fibril EM maps (https://www.ebi.ac.uk/emdb/).

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

AGGRESCAN3D server (https://biocomp.chem.uw.edu.pl/A3D2/) — structure-based aggregation prediction; CamSol (https://www-cohsoftware.ch.cam.ac.uk/index.php/camsolmethod) — solubility prediction; WALTZ-DB 2.0 (verify URL) — aggregation kinetics; GROMACS+PLUMED fibril simulation stack (https://github.com/gromacs/gromacs).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

GPU MARTINI CG-MD for large oligomerization systems; metadynamics enhanced sampling via PLUMED on GPU; GPU-trained GNN inference for sequence-based aggregation; CUDA-accelerated contact map tracking during aggregation. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
