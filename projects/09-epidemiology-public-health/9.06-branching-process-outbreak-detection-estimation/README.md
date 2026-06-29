# 9.6 — Branching-Process Outbreak Detection & Estimation

![difficulty](https://img.shields.io/badge/difficulty-Advanced-blue) ![maturity](https://img.shields.io/badge/maturity-Frontier%2FTheoretical-informational) ![domain](https://img.shields.io/badge/domain-Epidemiology%20%26%20Public%20Health-lightgrey)

> **🔴 Advanced · Frontier/Theoretical** — Domain 9: Epidemiology & Public Health · Catalog ID `9.6`
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

Models early epidemic growth as a Galton-Watson branching process or Hawkes point process to estimate the effective reproduction number Rt in near-real-time from case count time series. GPU parallelism enables simultaneous estimation of Rt across thousands of geographic units (counties, countries) simultaneously using batched Bayesian updates. The Hawkes process likelihood requires summing exponential kernels over all past events — a GPU-parallelised prefix sum operation. Branching process simulation (for outbreak probability calculations) is embarrassingly parallel: simulate 10⁵ independent outbreak realisations simultaneously on GPU to estimate extinction probabilities.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Galton-Watson branching process simulation, Hawkes self-exciting point process MLE, EpiEstim sliding-window Rt estimation, renewal equation Rt inference (Cori method), sequential Monte Carlo (particle filters) for real-time estimation, negative-binomial offspring distribution fitting, overdispersion estimation.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/branching-process-outbreak-detection-estimation.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/branching-process-outbreak-detection-estimation.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\branching-process-outbreak-detection-estimation.sln /p:Configuration=Release /p:Platform=x64
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

Catalog dataset notes: CDC FluView — weekly US influenza-like illness surveillance (https://www.cdc.gov/flu/weekly/) WHO Disease Outbreak News — global outbreak event data (https://www.who.int/emergencies/disease-outbreak-news) COVID-19 Data Repository (CSSE Johns Hopkins) — archived case/death time series (https://github.com/CSSEGISandData/COVID-19) ECDC Surveillance Atlas — European communicable disease surveillance (https://atlas.ecdc.europa.eu/)

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

EpiEstim (https://github.com/mrc-ide/EpiEstim) — R package for Rt estimation (CPU; GPU via batched extension) EpiNow2 (https://github.com/epiforecasts/EpiNow2) — Bayesian nowcasting and Rt estimation with Stan GPU backend tick (https://github.com/X-DataInitiative/tick) — GPU-accelerated Hawkes process learning library PyEpidemics (verify URL) — Python branching process simulation framework

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuRAND for Monte Carlo branching process simulation, Thrust parallel prefix sum for Hawkes likelihood, JAX/BlackJAX for GPU-based posterior inference; pattern: embarrassingly parallel ensemble simulation — each CUDA thread simulates one outbreak trajectory. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
