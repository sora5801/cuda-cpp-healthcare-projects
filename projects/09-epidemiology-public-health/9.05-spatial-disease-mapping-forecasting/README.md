# 9.5 — Spatial Disease Mapping & Forecasting

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Epidemiology%20%26%20Public%20Health-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 9: Epidemiology & Public Health · Catalog ID `9.5`
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

Estimates disease incidence surfaces and spatiotemporal risk across geographic grids using Bayesian geostatistical models (BYM, INLA, Gaussian Process regression). The Gaussian process kernel matrix computation scales as O(N²) in the number of spatial locations — for a 10k-pixel grid this is a 10⁸-element covariance matrix, whose Cholesky decomposition is dominated by GPU-accelerated dense linear algebra (cuBLAS). GPU-based MCMC samplers (BlackJAX on CUDA, Greta with GPU backend) achieve 380× speedup for epidemic forecasting models. Interpolating national case-counts to sub-district resolution using kriging is entirely parallelisable across prediction locations on GPU.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Besag-York-Mollié (BYM) spatial smoothing, Integrated Nested Laplace Approximation (INLA), Gaussian Process regression, kriging interpolation, spatiotemporal Kalman filtering, Bayesian hierarchical Poisson regression, neural ODE spatial models, ensemble Kalman filters.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/spatial-disease-mapping-forecasting.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/spatial-disease-mapping-forecasting.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\spatial-disease-mapping-forecasting.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: WHO Mortality Database — ICD-coded deaths by country and cause (https://www.who.int/data/data-collection-tools/who-mortality-database) IHME Global Burden of Disease — country-level disease incidence estimates (https://www.healthdata.org/gbd) CDC Wonder — US county-level disease surveillance data (https://wonder.cdc.gov/) NASA SEDAC Global Population Data — gridded population for exposure modelling (https://sedac.ciesin.columbia.edu/)

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

INLA / R-INLA (https://www.r-inla.org/) — fast Bayesian spatial modelling; GPU via PARDISO sparse solver BlackJAX (https://github.com/blackjax-devs/blackjax) — GPU-accelerated Bayesian sampling (HMC, NUTS) via JAX Greta (https://github.com/greta-dev/greta) — probabilistic programming with TensorFlow GPU backend for spatial models CARBayes (https://github.com/duncanplee/CARBayes) — R package for spatial Bayesian modelling (CPU; parallelisable)

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuBLAS for GP covariance matrix Cholesky, cuSPARSE for ICAR precision matrix operations, JAX XLA for GPU-accelerated MCMC; pattern: batch kriging over prediction grid points with fully GPU-resident covariance kernel. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
