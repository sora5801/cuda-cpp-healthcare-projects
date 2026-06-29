# 13.16 — Receptor Occupancy Imaging & PET Pharmacokinetics

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Pharmacology%20%26%20Clinical%20Quantitative%20Modeling-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 13: Pharmacology & Clinical Quantitative Modeling · Catalog ID `13.16`
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

Analyses PET neuroimaging data to quantify receptor occupancy by drug candidates across thousands of brain voxels simultaneously. The Logan reference tissue method and two-tissue compartmental models must be fitted to the time-activity curve (TAC) at each voxel — a problem with 100k+ independent nonlinear regressions that map directly to GPU parallelism. GPU-parallel voxel-wise model fitting achieves near-real-time analysis of 3D PET volumes (128×128×63 voxels). Virtual receptor occupancy simulations (coupling PBPK with brain RO submodel) for dose selection require batched ODE integration on GPU across candidate dose levels.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Logan reference tissue method, two-tissue compartmental model, simplified reference tissue model (SRTM), voxel-wise ODE fitting with Levenberg-Marquardt on GPU, Patlak graphical analysis, partial volume correction, kinetic parameter estimation (K1, k2, BP_ND).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/receptor-occupancy-imaging-pet-pharmacokinetics.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/receptor-occupancy-imaging-pet-pharmacokinetics.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\receptor-occupancy-imaging-pet-pharmacokinetics.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: OpenNeuro PET datasets — open-access brain PET with kinetic data (https://openneuro.org/) NeuroVault PET studies — aggregated neuroimaging PET data (https://neurovault.org/) BrainPET benchmark datasets (verify URL — NIMH) ADNI PET-amyloid data — longitudinal PET for Alzheimer imaging (https://adni.loni.usc.edu/)

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

NiftyPAD (verify URL) — GPU-parallelised PET kinetic modelling toolkit TPCCLIB (verify URL) — C library for PET kinetic analysis (CPU; GPU extension possible) Pumas (https://pumas.ai/) — GPU-accelerated brain RO-PBPK coupling in Julia SimplePET (https://github.com/UCL/simplicity) — Python PET simulation and analysis (verify URL)

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Custom CUDA Levenberg-Marquardt kernels for per-voxel TAC fitting, cuBLAS for covariance matrix inversion, cuFFT for PET sinogram reconstruction; pattern: one CUDA thread per voxel, embarrassingly parallel kinetic fitting. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
