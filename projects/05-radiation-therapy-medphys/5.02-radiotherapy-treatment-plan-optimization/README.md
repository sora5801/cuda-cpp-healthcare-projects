# 5.2 — Radiotherapy Treatment-Plan Optimization

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Radiation%20Therapy%20%26%20Medical%20Physics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 5: Radiation Therapy & Medical Physics · Catalog ID `5.2`
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

IMRT/VMAT plan optimization solves a large-scale constrained optimization: minimize dose to OARs subject to PTV coverage constraints, with variables being beam aperture shapes or fluence maps. The dose-influence matrix D (N_voxels × N_beamlets, typically 10⁶ × 10⁴) must be computed and stored on GPU; the iterative optimizer (gradient descent, IPOPT, L-BFGS) performs repeated sparse matrix-vector products (D·x) per iteration. GPU SpMV reduces each DMAT-vector product from seconds to milliseconds, enabling real-time adaptive re-optimization. Biological-effect optimization (TCP/NTCP) and robust optimization over uncertainty scenarios further multiply the compute by the number of scenarios (~50–100 for robust plans).

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Fluence-map optimization (quadratic programming, L-BFGS), direct aperture optimization (DAO), volumetric modulated arc therapy (VMAT) optimization, robust optimization (minimax), biological TCP/NTCP optimization, multi-criteria optimization (Pareto front navigation), deep learning dose prediction (U-Net).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/radiotherapy-treatment-plan-optimization.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/radiotherapy-treatment-plan-optimization.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\radiotherapy-treatment-plan-optimization.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: OpenKBP (knowledge-based planning) dataset (https://github.com/ababier/open-kbp) — 340 head-and-neck IMRT plans; TCIA RT datasets; PlanIQ (verify URL); AAPM TG-263 structure naming dataset; OpenTPS test datasets.

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

matRad (https://github.com/e0404/matRad) — open-source MATLAB treatment planning, photon/proton/carbon; pyRadPlan (https://github.com/e0404/pyRadPlan) — Python interoperable extension of matRad; CERR (https://github.com/cerr/CERR) — MATLAB comprehensive RT research platform with DICOM-RT; OpenTPS (https://opentps.org/) — open-source Python/GPU treatment planning system (verify URL).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuSPARSE (SpMV for D·fluence products); cuBLAS (OAR/PTV dose-volume histogram computation); CUDA warp-level reductions for DVH statistics; GPU-resident D-matrix in CSR format; multi-GPU for scenario-parallel robust optimization. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
