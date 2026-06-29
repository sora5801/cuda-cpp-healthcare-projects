# 10.2 — Real-Time Soft-Tissue Deformation for Surgical Simulation

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Biomechanics%2C%20Biomedical%20Devices%20%26%20Surgery-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 10: Biomechanics, Biomedical Devices & Surgery · Catalog ID `10.2`
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

Surgical simulators require sub-10 ms deformation updates on organ meshes of tens to hundreds of thousands of elements so that haptic devices can deliver force feedback without perceived lag. Position-Based Dynamics (PBD) and its extended variant XPBD run all constraint projections in parallel, with each particle or constraint mapped to a CUDA thread. The 2024 dissection simulator demonstrated real-time performance on >100 K particles, including topological cuts, using parallelized graph-based shape matching on GPU. Material Point Method (MPM) on GPU further handles cutting and tearing by decoupling Eulerian background grids from Lagrangian particles. Hybrid organ models combining rigid bones with deformable soft tissue use adaptive octree refinement on GPU to concentrate resolution near contact zones.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Position-Based Dynamics (PBD/XPBD), Total Lagrangian Explicit Dynamics (TLED), graph-based shape matching, Material Point Method (MPM), corotational linear FEM, multigrid preconditioned conjugate gradient, near-second-order Jacobi/Gauss-Seidel elastodynamics (JGS2).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/real-time-soft-tissue-deformation-for-surgical-simulation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/real-time-soft-tissue-deformation-for-surgical-simulation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\real-time-soft-tissue-deformation-for-surgical-simulation.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: SOFA Framework benchmark scenes — laparoscopic and open-surgery deformable organ models (https://www.sofa-framework.org/); Kaggle Liver CT Segmentation — 3D liver meshes for deformation benchmarking (https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation); MRI Breast Tissue Segmentation (nnU-Net preprocessed) for biomechanical modeling (https://arxiv.org/abs/2411.18784); iMSTK Test Suite — pre-built surgical scenario meshes (https://www.imstk.org/).

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

SOFA Framework (https://github.com/sofa-framework/sofa) — open-source physics engine with GPU PBD plugins and haptic coupling; iMSTK (https://github.com/Kitware/iMSTK) — interactive medical simulation toolkit with CUDA deformation; NVIDIA FleX (https://github.com/NVIDIAGameWorks/FleX) — GPU PBD particle solver adapted for surgical contexts; CRESSim-MPM (verify URL, search "CRESSim MPM surgical simulation GPU") — GPU MPM library for cutting/suturing simulation.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

CUDA kernels for per-constraint projection (one thread per constraint in parallel Gauss-Seidel with graph coloring), Thrust for particle neighbor search, cuSPARSE for global stiffness assembly; pattern: coloring-based Gauss-Seidel to avoid write conflicts → warp-shuffle reductions for constraint residuals → atomic updates on shared boundary nodes. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
