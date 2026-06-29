# 1.13 — Pharmacophore & 3D Shape Screening

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Drug%20Discovery%20%26%20Molecular%20Design-lightgrey)

> **🟢 Beginner · Established** — Domain 1: Drug Discovery & Molecular Design · Catalog ID `1.13`
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

Pharmacophore and shape-based screening compares 3D query features (hydrogen bond donors/acceptors, hydrophobic regions, ionizable groups, molecular shape) against library conformers, capturing complementarity not encoded in 2D fingerprints. ROCS (OpenEye) uses a volumetric Gaussian overlap function (ShapeTanimoto + ColorTanimoto) that is differentiable and GPU-friendly. Screening billions of conformers requires GPU-parallel overlap computation across independent molecule pairs. This is a key pre-filtering step before docking in virtual screening pipelines.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Gaussian volume overlap (Tversky/Tanimoto), Fast Overlay of Chemical Structures (FOCS), pharmacophore feature matching (HBD/HBA/hydrophobic/aromatic), conformer ensemble generation, rigid body alignment (quaternion-based).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/pharmacophore-3d-shape-screening.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/pharmacophore-3d-shape-screening.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\pharmacophore-3d-shape-screening.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: ZINC20 conformer libraries (https://zinc20.docking.org); DUD-E (https://dude.docking.org); Enamine REAL conformer sets (https://enamine.net); Directory of Useful Decoys-Enhanced including 3D conformers (verify URL).

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

ROCS (OpenEye/Cadence) — commercial GPU 3D shape screening (https://www.eyesopen.com/rocs); Open3DQSAR (https://open3dqsar.sourceforge.io) — open 3D-QSAR tool; RDKit shape tools (https://github.com/rdkit/rdkit) — open Gaussian overlap via PyTorch extension (verify URL); Pharmer (https://github.com/dkoes/pharmer) — open pharmacophore search tool.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Warp-parallel Gaussian overlap evaluation over conformer pairs; texture memory for pre-computed atom volumes; GPU-batched rigid-body alignment using quaternion representation; cuBLAS for rotation matrix applications. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
