# 10.3 — Implant & Prosthetic Design Optimization

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Biomechanics%2C%20Biomedical%20Devices%20%26%20Surgery-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 10: Biomechanics, Biomedical Devices & Surgery · Catalog ID `10.3`
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

Patient-specific implants (hip, knee, spinal, dental) require iterative structural optimization over high-resolution 3D voxel grids (>1 M elements), where density or level-set fields evolve based on sensitivity analysis from repeated FEA solves. GPU acceleration makes three-dimensional SIMP (Solid Isotropic Material with Penalization) topology optimization tractable: a single density update pass over a 256³ grid requires ~16 M stiffness evaluations that execute in parallel. Lattice-structure implants for osseointegration require multiscale homogenization, computing effective elastic tensors for thousands of unit-cell configurations in parallel on GPU. Bone-remodeling feedback loops then validate implant geometry by simulating load transfer over years of use.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

SIMP topology optimization, density-based level-set method, homogenization of periodic lattices, finite-element sensitivity analysis, optimality criteria (OC) update, bone-remodeling (Weinans/Beaupré) adaptation, multi-objective Pareto optimization.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/implant-prosthetic-design-optimization.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/implant-prosthetic-design-optimization.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\implant-prosthetic-design-optimization.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: OrthoLoad Implant Loading Database — in vivo hip/knee/spine implant force telemetry (https://orthoload.com/); MICCAI 2023 VerSe Challenge — vertebral shape dataset for spinal implant design (https://verse-challenge.github.io/); Hip Implant Topology Dataset — validated micro-FE lattice endoprostheses (see https://www.nature.com/articles/s41598-024-56327-4); FDA Orthopaedic Simulator Database — standardized fatigue loading profiles (verify URL via FDA.gov).

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

GPU-Accelerated Topology Optimization (Paulino group, Princeton) (https://paulino.princeton.edu/journal_papers/2013/SMO_13_TowardGPUAccelerated.pdf) — multigrid GPU SIMP reference implementation; Simple and Efficient GPU TO (https://www.sciencedirect.com/science/article/pii/S0045782523001676) — open-source GPU TO code from 2023 CMAME paper (verify repo link in supplementary); FEBio (https://github.com/febiosoftware/FEBio) — sensitivity analysis infrastructure; ToPy (https://github.com/williamhunter/topy) — Python 2D/3D TO (CPU, GPU-extensible reference).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuSPARSE for repeated sparse FE solves, cuDNN for CNN-based TO surrogate acceleration, Thrust for parallel density-filter convolutions; pattern: element-parallel stiffness + sensitivity computation → parallel density update → GPU multigrid V-cycle for equilibrium solve. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
