# 9.9 — Hospital Capacity & Surge Demand Forecasting

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Epidemiology%20%26%20Public%20Health-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 9: Epidemiology & Public Health · Catalog ID `9.9`
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

Predicts short-term hospital admission volumes, ICU occupancy, and ventilator demand to enable proactive resource allocation during epidemic surges or seasonal peaks. GPU-accelerated LSTM, Transformer, and ensemble models trained on EHR admission records, regional case counts, wastewater signals, and mobility data produce rolling 14-day forecasts. The volume of hospital time series (thousands of hospitals × dozens of admission types × 365 days/year) is processed in parallel on GPU; each hospital's time series is a separate batch element. Real-time retraining on streaming data requires frequent mini-batch SGD on GPU to adapt to evolving epidemic waves.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

LSTM/GRU multi-step forecasting, Temporal Fusion Transformers, N-BEATS, Prophet (Bayesian decomposition), Gaussian process regression for uncertainty, hierarchical reconciliation (MinT), ensemble averaging, ARIMA + neural hybrids, conformal prediction intervals.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/hospital-capacity-surge-demand-forecasting.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/hospital-capacity-surge-demand-forecasting.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\hospital-capacity-surge-demand-forecasting.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: HHS Protect Hospital Capacity Data — US hospital capacity and admissions (https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/6xf2-c3ie) ECDC Hospital Data — European hospital admissions and ICU occupancy (https://www.ecdc.europa.eu/en/covid-19/data) NHS England Situation Reports — UK hospital admissions and bed occupancy (https://www.england.nhs.uk/statistics/) COVID-19 Forecast Hub submissions — ensemble of >50 models (https://covid19forecasthub.org/)

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

PyTorch-Forecasting (https://github.com/jdb78/pytorch-forecasting) — TFT, LSTM, N-BEATS on GPU Darts (https://github.com/unit8co/darts) — multi-model time series forecasting with GPU backend COVID-19 Forecast Hub (https://github.com/reichlab/covid19-forecast-hub) — ensemble model aggregation infrastructure GluonTS (https://github.com/awslabs/gluonts) — probabilistic time series on GPU via MXNet/PyTorch

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuDNN for temporal model training, JAX XLA for parallelised Gaussian process forecasting, NCCL for multi-GPU ensemble training; pattern: panel data parallel — each hospital's time series as a batch element. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
