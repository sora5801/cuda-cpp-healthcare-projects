# 6.18 — ECG Forward Problem & Body-Surface Potential Mapping

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟢 Beginner · Established** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.18`
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

The ECG forward problem maps cardiac electrical sources (transmembrane currents from EP simulation) to body-surface potentials via the quasi-static Poisson equation on a torso volume conductor model. The transfer matrix (lead-field matrix) is computed once by solving many FEM boundary value problems (one per electrode), then applied repeatedly as a dense matrix-vector product at each time step of the EP simulation. GPU acceleration is ideal for both the batched FEM assembly and the dense matrix-vector multiply.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Quasi-static Poisson equation (torso conductivity model), finite element method on torso mesh, lead-field/transfer matrix computation, multipole source representation, method of fundamental solutions, ECG inverse problem (regularized Tikhonov, total variation), boundary element method (BEM).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/ecg-forward-problem-body-surface-potential-mapping.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/ecg-forward-problem-body-surface-potential-mapping.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\ecg-forward-problem-body-surface-potential-mapping.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: PhysioNet ECG databases (https://physionet.org); EDGAR body-surface potential database (https://edgar.sci.utah.edu — verify URL); Cardioid ECG module examples (https://github.com/llnl/cardioid); Visible Human torso geometry (https://www.nlm.nih.gov/research/visible/visible_human.html).

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

Cardioid/LLNL (https://github.com/llnl/cardioid) — includes ECG forward solver module; openCARP (https://git.opencarp.org/openCARP/openCARP) — ECG lead calculation post-processing; SCIRun (https://github.com/SCIInstitute/SCIRun) — Utah scientific computing platform for ECG forward/inverse; APBS (https://github.com/Electrostatics/apbs) — electrostatics PDE solver adaptable to torso geometry.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuBLAS DGEMV for transfer-matrix application at each time step; cuSOLVER for FEM system solve during transfer-matrix construction; batched cuSOLVER for simultaneous electrode-source BVPs; pattern: parallel BVP solves (one per electrode) with shared torso mesh. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
