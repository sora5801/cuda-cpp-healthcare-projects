# 1.14 — Conformer Ensemble Generation

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟢 Beginner · Established** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.14`
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

Drug-like molecules are flexible; binding-relevant conformers must be generated before 3D screening or docking. RDKit ETKDG embeds molecules in 3D using experimental torsion knowledge (ETKDGv3) and distance geometry; generation of thousands of conformers per molecule for a library of millions is a CPU bottleneck. GPU acceleration is achieved by batching conformer embedding across many molecules simultaneously. Alternatively, ML-based conformer predictors (OMEGA-ML, GeoMol, TorsionalDiffusion) use GPU neural networks trained on crystallographic torsion distributions.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Experimental torsion-angle knowledge distance geometry (ETKDG), MMFF94/UFF energy minimization, breadth-first conformer pruning (RMSD clustering), torsional diffusion (ML), graph neural network conformer prediction.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/conformer-ensemble-generation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/conformer-ensemble-generation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\conformer-ensemble-generation.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: GEOM — 37M conformers of drug-like molecules with DFT energies (https://github.com/learningmatter-mit/geom); CSD torsion library (https://www.ccdc.cam.ac.uk); COD (Crystallography Open Database) — crystal structures for torsion validation (https://www.crystallography.net); PDB small molecule conformations (https://www.rcsb.org).

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

RDKit ETKDG (https://github.com/rdkit/rdkit) — standard conformer engine, GPU-batched via RDKit-GPU (verify URL); TorsionalDiffusion (https://github.com/gcorso/torsional-diffusion) — GPU diffusion model for conformer sampling; GeoMol (https://github.com/PattanaikL/GeoMol) — ML conformer prediction; Frog2 / OMEGA (OpenEye, commercial) — fast conformer generators.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Batched SVD/distance geometry on GPU via cuSOLVER; custom CUDA kernels for pairwise RMSD computation; GPU-parallel MMFF energy minimization via molecular gradient descent; PyTorch-based diffusion inference with CUDA tensors. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
