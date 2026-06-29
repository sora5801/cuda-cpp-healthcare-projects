# 6.27 — Parameter Estimation & Data Assimilation for Physiological Models

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Computational%20Physiology%20%26%20Systems%20Biology-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 6: Computational Physiology & Systems Biology · Catalog ID `6.27`
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

Fitting ODE/PDE physiological models to patient-specific clinical data (ECG, pressure waveforms, biomarker time series) requires repeated forward simulation within an optimization or Bayesian inference loop. Ensemble Kalman filters (EnKF) update an ensemble of 50–500 model states in parallel with incoming observations; unscented Kalman filters (UKF) propagate 2N+1 sigma points. GPU acceleration of the forward model ensemble is the bottleneck.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Ensemble Kalman filter (EnKF), unscented Kalman filter (UKF), particle filter (sequential Monte Carlo), adjoint-based gradient optimization (L-BFGS), variational data assimilation (4D-Var), Gaussian process emulator surrogate, Bayesian optimization, trust-region methods.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/parameter-estimation-data-assimilation-for-physiological-models.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/parameter-estimation-data-assimilation-for-physiological-models.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\parameter-estimation-data-assimilation-for-physiological-models.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: PhysioNet MIMIC clinical waveforms (https://physionet.org); UK Biobank cardiac functional parameters (https://www.ukbiobank.ac.uk); Zenodo cardiac mechanics emulation dataset (https://zenodo.org/records/7075055); openCARP community experiments (https://opencarp.org/community/community-experiments).

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

SUNDIALS/CVODES (https://github.com/LLNL/sundials) — sensitivity-aware ODE integrator for adjoint gradient; simcardems (https://github.com/ComputationalPhysiology/simcardems) — cardiac twin with parameter fitting; SALib (https://github.com/SALib/SALib) — sensitivity analysis for parameter prioritization; PyMC (https://github.com/pymc-devs/pymc) — probabilistic programming with GPU via JAX/Aesara backend.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Batch forward ODE on GPU (ensemble members); cuBLAS for EnKF covariance update (N×N matrix operations); cuSOLVER for Kalman gain; CUDA Thrust for particle resampling; pattern: ensemble-parallel forward solves + host-side EnKF analysis step. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
