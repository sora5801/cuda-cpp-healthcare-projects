# 10.1 — FEA of Bone & Tissue

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Biomechanics%2C%20Biomedical%20Devices%20%26%20Surgery-lightgrey)

> **🟢 Beginner · Established** — Domain 10: Biomechanics, Biomedical Devices & Surgery · Catalog ID `10.1`
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

Finite-element analysis of bone and soft tissue solves systems of millions of coupled equations relating stress, strain, and material nonlinearity under physiological loading. GPU parallelism targets the sparse-matrix assembly and iterative linear-solver (conjugate-gradient or multigrid) phases, which dominate wall time in large 3D meshes. Co-rotational and total-Lagrangian explicit dynamics (TLED) formulations map naturally to SIMT execution because each element's stiffness update is independent. Bone-remodeling simulations (Wolff's law) couple mechanical fields with density update rules, requiring repeated solve-update-resolve cycles that each benefit from CUDA acceleration. Real-world targets include vertebral fracture prediction, hip-implant stress-shielding, and micro-CT-derived trabecular models with >10 M elements.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Total Lagrangian Explicit Dynamics (TLED), co-rotational FEM, neo-Hookean / Mooney-Rivlin hyperelasticity, preconditioned conjugate gradient (PCG) with Jacobi or incomplete-Cholesky preconditioners, bone-remodeling (Beaupré–Carter) adaptation loops.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/fea-of-bone-tissue.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/fea-of-bone-tissue.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\fea-of-bone-tissue.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: FEBio Benchmark Suite — verified test problems for nonlinear biomechanical FEA (https://febio.org/knowledgebase/); Open Knee(s) — subject-specific knee joint FE models with segmented cartilage/bone (https://simtk.org/projects/openknee); Visible Human Project — full CT/MRI cadaver data for mesh generation (https://www.nlm.nih.gov/research/visible/visible_human.html); Bone-Load Database (Bergmann et al.) — in vivo implant load telemetry for hip and knee (https://orthoload.com/).

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

FEBio (https://github.com/febiosoftware/FEBio) — open-source nonlinear FE solver for biomechanics, C++, with GPU-solver hooks; NiftySim (https://github.com/eloygarcia/niftysim) — CUDA TLED soft-tissue FE toolkit from UCL; NVIDIA CUDALibrarySamples (https://github.com/NVIDIA/CUDALibrarySamples) — cuSPARSE/cuSolver conjugate-gradient templates; Awesome-Biomechanics (https://github.com/modenaxe/awesome-biomechanics) — curated dataset/tool index.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuSPARSE (SpMV in PCG inner loop), cuSolver (direct sparse factorization), cuBLAS (dense BLAS), Thrust (parallel reductions); pattern: one CUDA thread per element for stiffness assembly → global atomic scatter into CSR matrix → iterative solver in cuSPARSE. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
