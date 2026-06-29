# 6.2 — Whole-Heart Digital Twin

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.2`
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

Integrates patient-specific cardiac geometry (from CMR segmentation), fiber orientation (rule-based or DTI), EP simulation, and mechanical contraction into a unified virtual organ calibrated to clinical measurements. Building the twin requires iterative parameter estimation loops—thousands of forward simulations of the EP+mechanics PDE system—making GPU acceleration critical not just for each simulation but for the ensemble inference step. Differentiable simulators (e.g., TorchCor) allow gradient-based parameter fitting through the forward model. Hemodynamic boundary conditions couple the twin to a lumped Windkessel circulation model.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Bidomain/monodomain EP, active-strain / active-stress cardiac mechanics (nonlinear elasticity), Windkessel 3-element lumped circulation, rule-based fiber assignment (Bayer-Blake-Plank), adjoint-based or ensemble Kalman filter parameter estimation, finite element method (FEM) with tetrahedral meshes.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/whole-heart-digital-twin.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/whole-heart-digital-twin.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\whole-heart-digital-twin.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: UK Biobank Cardiac MRI — 100 000+ cine CMR (https://www.ukbiobank.ac.uk); Zenodo Synthetic Biventricular Heart Meshes — 1 000 virtual cohort meshes (https://zenodo.org/records/4506930); Visible Human Project — full-body cryosection + CT + MRI (https://www.nlm.nih.gov/research/visible/visible_human.html); ACDC MICCAI — 100-patient CMR segmentations (https://www.creatis.insa-lyon.fr/Challenge/acdc/).

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

openCARP (https://git.opencarp.org/openCARP/openCARP) — EP component of twins; simcardems (https://github.com/ComputationalPhysiology/simcardems) — FEniCS-based cardiac electromechanics coupling; TorchCor (https://github.com/sagebei/torchcor) — PyTorch GPU cardiac EP FEM for differentiable twin fitting; Awesome-Cardiac-Digital-Twins list (https://github.com/lileitech/Awesome-Cardiac-Digital-Twins) — curated resource index.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuSPARSE + cuSOLVER (FEM assembly/solve), cuBLAS (adjoint vector ops), custom CUDA kernels (ionic ODE batch); pattern: batched forward solves across ensemble members for parameter inference; mixed precision (FP16 forward, FP32 gradient accumulation). --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
