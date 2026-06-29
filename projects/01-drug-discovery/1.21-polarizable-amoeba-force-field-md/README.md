# 1.21 — Polarizable / AMOEBA Force Field MD

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.21`
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

Classical fixed-charge force fields miss polarization effects crucial for accurate binding free energies and ionic interactions. The AMOEBA force field includes point multipoles (up to quadrupoles) and induced dipoles solved self-consistently at each MD step via an iterative solver (conjugate gradient). This increases cost ~10× over AMBER but GPU implementation in Tinker-HP achieves >200-fold speedup over single-CPU, making microsecond AMOEBA simulations of large proteins feasible. Applications include protein-ligand FEP with AMOEBA and pKa prediction in complex electrostatic environments.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Induced dipole iteration (conjugate gradient), Ewald summation for multipoles (PME-multipole), AMOEBA water model, HIPPO force field, PIMD with polarizable FF.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/polarizable-amoeba-force-field-md.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/polarizable-amoeba-force-field-md.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\polarizable-amoeba-force-field-md.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: AMOEBA protein force field parameter files (https://github.com/TinkerTools/tinker); WaterMap/hydration site datasets (Schrodinger, verify URL); BindingDB experimental affinities (https://www.bindingdb.org); NIST thermophysical properties (https://webbook.nist.gov).

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

Tinker-HP (https://github.com/TinkerTools/tinker-hp) — massively parallel GPU AMOEBA MD; OpenMM AMOEBA plugin (https://github.com/openmm/openmm) — AMOEBA on CUDA; Tinker9 (https://github.com/TinkerTools/tinker9) — GPU-native Tinker rewrite; AMOEBA+ FF parameters (https://github.com/TinkerTools/poltype2).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom CUDA conjugate-gradient solver for induced dipoles; cuFFT for multipole PME; warp-synchronous reduction for energy accumulation; multi-GPU via MPI domain decomposition with NCCL. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
