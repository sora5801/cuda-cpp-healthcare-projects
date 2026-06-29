# 9.2 — Large-Scale Compartmental & Metapopulation Models

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Epidemiology%20%26%20Public%20Health-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 9: Epidemiology & Public Health · Catalog ID `9.2`
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

Solves large systems of ODEs or stochastic differential equations (SDEs) describing disease dynamics across thousands of geographic patches interconnected by mobility flows (SIR at metapopulation scale, seasonal forcing, age structure). ODE integration over thousands of patches with coupling matrices is equivalent to a batched sparse matrix-vector multiply at each time step — a cuSPARSE-accelerated operation. Monte Carlo uncertainty quantification requires thousands of independent ODE solves in parallel on GPU, each with different parameter samples. GPU-based adaptive stepsize RK4/5 solvers (Torchdiffeq's `dopri5` on GPU) handle stiff biological dynamics efficiently.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Runge-Kutta 4/5 ODE integration on GPU, tau-leaping for stochastic compartmental models, MCMC parameter inference (ensemble MCMC), Approximate Bayesian Computation (ABC), metapopulation coupling via mobility matrix, seasonal forcing with Fourier series, age-structured SEIR with contact matrices.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/large-scale-compartmental-metapopulation-models.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/large-scale-compartmental-metapopulation-models.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\large-scale-compartmental-metapopulation-models.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: GLEAM — global airline + commuting network for metapopulation coupling (https://www.gleamviz.org/) WHO Weekly Epidemiological Reports — case counts for parameter calibration (https://www.who.int/emergencies/situations) CDC FluView — US influenza surveillance by week and region (https://www.cdc.gov/flu/weekly/) COVID-19 Data Repository by CSSE at Johns Hopkins (archived) — global case/death time series (https://github.com/CSSEGISandData/COVID-19)

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

Epiflows / EpiModel (https://github.com/EpiModel/EpiModel) — network-based compartmental modelling in R Torchdiffeq (https://github.com/rtqichen/torchdiffeq) — GPU-accelerated neural ODE and standard ODE solvers MEmilio (https://github.com/SciCompMod/memilio) — high-performance C++/CUDA epidemic simulation PyGOM (https://github.com/ukhsa-collaboration/pygom) — Python compartmental ODE modelling framework

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuSPARSE for mobility matrix coupling, cuRAND for stochastic tau-leaping, custom RK4 CUDA kernel for parallel ODE batch; pattern: each CUDA thread block integrates one metapopulation patch ODE system, with shared memory holding coupling matrices. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
