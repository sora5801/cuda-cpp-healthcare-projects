# 9.8 — Wastewater-Based Epidemiology & Signal Detection

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Epidemiology%20%26%20Public%20Health-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 9: Epidemiology & Public Health · Catalog ID `9.8`
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

Infers community-level pathogen prevalence from viral RNA concentrations in wastewater, combining RT-qPCR signal time series with meteorological, demographic, and mobility covariates to nowcast and forecast disease incidence. GPU-accelerated deep learning (LSTM, Temporal Fusion Transformers) processes multivariate time series from thousands of sampling sites simultaneously; the data dimensionality is high (dozens of wastewater markers × weather variables × mobility indices per site). Bayesian hierarchical models fitted on GPU (via Stan with GPU backend or JAX) account for spatial correlation across sewage catchments. Deconvolution of wastewater signal to estimate case counts involves non-negative least-squares problems solved in parallel across sites.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Non-negative least-squares deconvolution, LSTM/GRU time series prediction, Temporal Fusion Transformers (TFT), Bayesian hierarchical regression, anomaly detection (isolation forests, CUSUM control charts), Poisson regression for count outcomes, spatial kriging for site interpolation.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/wastewater-based-epidemiology-signal-detection.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/wastewater-based-epidemiology-signal-detection.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\wastewater-based-epidemiology-signal-detection.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: NWSS (National Wastewater Surveillance System) — US wastewater SARS-CoV-2 and flu data (https://www.cdc.gov/nwss/) EU Sewage Sentinel System for SARS-CoV-2 (verify URL) — European wastewater surveillance WastewaterSCAN — Stanford-led multi-pathogen wastewater monitoring (https://www.wastewaterscan.org/) OpenWastewaterData (verify URL) — aggregated global wastewater surveillance

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

PyTorch-Forecasting (https://github.com/jdb78/pytorch-forecasting) — TFT and LSTM for multivariate time series on GPU Pyro (https://github.com/pyro-ppl/pyro) — GPU probabilistic programming for Bayesian wastewater signal deconvolution Darts (https://github.com/unit8co/darts) — time series forecasting library with GPU support NWSS Data Dashboard tools (https://www.cdc.gov/nwss/wastewater-surveillance-data-reporting.html) — CDC reference implementation

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuDNN for temporal model training, Pyro ELBO optimisation on GPU, cuBLAS for deconvolution least-squares; pattern: data-parallel forecasting across thousands of wastewater sites on GPU. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
