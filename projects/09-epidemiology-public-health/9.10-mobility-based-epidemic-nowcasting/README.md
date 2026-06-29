# 9.10 — Mobility-Based Epidemic Nowcasting

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Epidemiology%20%26%20Public%20Health-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 9: Epidemiology & Public Health · Catalog ID `9.10`
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

Infers current epidemic state and short-term trajectory from human mobility data (mobile phone GPS, retail foot traffic, transit ridership) using data assimilation methods that combine mobility signals with epidemiological models. GPU enables rapid sequential Monte Carlo (particle filter) updates as new mobility observations arrive hourly, running thousands of particles simultaneously. Graph neural networks learn spatial transmission patterns from mobility flow matrices — a GPU-parallelised sparse graph convolution. The bottleneck is the batched epidemic ODE integration for all particles in the ensemble simultaneously.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Sequential Monte Carlo (particle filtering), ensemble Kalman filter (EnKF), graph convolutional networks on mobility graphs, LSTM encoder-decoder for mobility sequence learning, MAP estimation for transmission rate, community mobility indices as predictors (Google CMR).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/mobility-based-epidemic-nowcasting.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/mobility-based-epidemic-nowcasting.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\mobility-based-epidemic-nowcasting.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: Google Community Mobility Reports — country/region mobility indices during COVID-19 (https://www.google.com/covid19/mobility/) SafeGraph/Dewey POI visit data — US retail foot traffic (verify access terms) Apple Mobility Trends — routing request data by transit type (verify URL) Citymapper Mobility Index — urban mobility across 40 cities (verify URL)

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

GLEAM mobility pipeline (https://www.gleamviz.org/) — global airline + commuting mobility for epidemic modelling CuPy (https://github.com/cupy/cupy) — GPU NumPy for particle filter implementation Epiforecast (verify URL) — real-time epidemic nowcasting framework PYMC (https://github.com/pymc-devs/pymc) — probabilistic programming with GPU JAX/Numba backend for data assimilation

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuRAND for particle resampling, cuBLAS for ensemble matrix operations, cuGraph for mobility graph convolutions; pattern: particle filter with GPU-parallel ODE integration and resampling. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
