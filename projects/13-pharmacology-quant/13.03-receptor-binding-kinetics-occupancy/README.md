# 13.3 — Receptor Binding Kinetics & Occupancy

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Pharmacology%20%26%20Clinical%20Quantitative%20Modeling-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 13: Pharmacology & Clinical Quantitative Modeling · Catalog ID `13.3`
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

Simulates drug-receptor association, dissociation, and signalling downstream of receptor occupancy using differential equation models (two-state, ternary complex, operational models of agonism). In receptor occupancy (RO) imaging data analysis, GPU parallelism enables simultaneous fitting of PET tracer binding across thousands of brain voxels. For in silico virtual screening, GPU batch evaluation of binding kinetics models for thousands of drug candidates (each with different kon/koff) is the bottleneck — solved with CUDA-batched ODE integration. Extended kinetic models (induced-fit docking, conformational selection) couple binding kinetics to structural biology force fields for GPU-accelerated MD-enhanced occupancy predictions.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Two-state receptor model ODE, Ternary Complex Model (TCM), Operational Model of Agonism, kinetic rate equation fitting (kon, koff, Kd), PET Logan reference method, Receptor Occupancy ED50 estimation, cAMP/calcium signalling cascade ODEs, mean-field receptor population models.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/receptor-binding-kinetics-occupancy.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/receptor-binding-kinetics-occupancy.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\receptor-binding-kinetics-occupancy.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: ChEMBL binding kinetics data — kon/koff/Kd for thousands of drug-receptor pairs (https://www.ebi.ac.uk/chembl/) BindingDB kinetics subset (https://www.bindingdb.org/) OpenNeuro PET datasets — receptor occupancy imaging data (https://openneuro.org/) Guide to Pharmacology (GtoPdb) — curated receptor/ligand database (https://www.guidetopharmacology.org/)

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

PyDyNo (verify URL) — dynamic receptor simulation in Python RTKI (Receptor-Target Kinetics Interface) (verify URL) — kinetics fitting framework nvQSP (https://github.com/NVIDIA-Digital-Bio/nvQSP) — GPU ODE batching applicable to receptor kinetics models PySB (https://github.com/pysb/pysb) — Python rule-based biochemical network modelling

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom CUDA RK4 batched ODE kernels for receptor kinetics, cuRAND for parameter uncertainty propagation, cuBLAS for Jacobian computation; pattern: one CUDA thread per drug candidate, each solving receptor binding ODEs in parallel. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
