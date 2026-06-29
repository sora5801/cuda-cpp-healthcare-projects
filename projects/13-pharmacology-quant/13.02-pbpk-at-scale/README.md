# 13.02 — PBPK at Scale

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Pharmacology%20%26%20Clinical%20Quantitative%20Modeling-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 13: Pharmacology & Clinical Quantitative Modeling · Catalog ID `13.02`
>
> _Educational only — not for clinical/dosing use (see CLAUDE.md §8)._

## Summary

Simulate a **virtual population** pharmacokinetic study: solve a physiologically-
based pharmacokinetic (PBPK) ODE for thousands of virtual patients — each with
different sampled physiology — and summarize each one's drug exposure (Cmax, Tmax,
AUC). Every patient is an independent ODE solve, so each GPU thread integrates one
patient with RK4. This is the ensemble-ODE pattern (cf. `9.02`) applied to a
multi-compartment PK model with per-patient parameter sampling.

## What this computes & why the GPU helps

PBPK models drug disposition through interconnected physiological compartments
(here a teaching reduction: gut depot → central/plasma → peripheral tissue, with
absorption `ka`, clearance `CL`, and inter-compartment flow `Q`). High-throughput
screening solves the system for thousands of compounds × hundreds–thousands of
virtual subjects — 10⁴–10⁶ simultaneous ODE solves, ideal for GPU-parallel
Runge-Kutta (NVIDIA's nvQSP). The solves are independent → one thread per patient.

**The parallelized work** is the ensemble of RK4 integrations; each thread runs a
full time loop in registers and writes Cmax/Tmax/AUC.

## The algorithm in brief

- **Sample** each patient's `ka, CL, Vc, Vp, Q` log-normally around population medians.
- **Integrate** the 3-compartment ODE with RK4 (oral dose into the gut depot).
- **Summarize**: Cmax (peak plasma conc), Tmax (time of peak), AUC (trapezoidal).

See [THEORY.md](THEORY.md) for the model, RK4, and the full ~15-compartment PBPK.

## Build

Requires **Visual Studio 2026** (v145) + **CUDA 13.3** ([docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/pbpk-at-scale.sln`.
2. **`Release|x64`** → **Build** → `build/x64/Release/pbpk-at-scale.exe`.

CLI: `msbuild build\pbpk-at-scale.sln /p:Configuration=Release /p:Platform=x64`

## Run the demo

```powershell
./demo/run_demo.ps1
```

Integrates the virtual population on CPU + GPU and verifies the metrics match.

## Data

- **Sample (committed):** `data/sample/pbpk_params.txt` — the population config.
- **Realistic PBPK:** PK-Sim / nvQSP (whole-body, ~15 compartments) — see
  `scripts/download_data.ps1` and [data/README.md](data/README.md).
- Bigger population: `python scripts/make_synthetic.py --patients 100000`.

## Expected output

`demo/expected_output.txt` holds the deterministic sample patients and population
summary. The GPU (`src/kernels.cu`) and CPU (`src/reference_cpu.cpp`) share the
model + RK4 + RNG (`src/pbpk.h`) in **double precision**, so per-patient metrics
agree to ~machine precision (`worst diff ≈ 1e-14`). The mean AUC ≈ dose/CL — the
standard PK identity — confirming the model.

## Code tour

1. [`src/main.cu`](src/main.cu) — load, CPU + GPU integrate, verify, print samples + summary.
2. [`src/pbpk.h`](src/pbpk.h) — **the PBPK ODE + RK4 + per-patient sampling** (host + device).
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU population interface (one thread per patient).
4. [`src/kernels.cu`](src/kernels.cu) — the population kernel + host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the serial reference.

## Prior art & further reading

- **PK-Sim** (<https://github.com/Open-Systems-Pharmacology/PK-Sim>) — whole-body PBPK software.
- **nvQSP** (<https://github.com/NVIDIA-Digital-Bio/nvQSP>) — NVIDIA GPU QSP/PBPK ODE solvers (RODAS4).
- Rowland & Tozer, *Clinical Pharmacokinetics and Pharmacodynamics* — PK fundamentals.

Study these for production modelling; reimplement the pattern didactically (CLAUDE.md §2).

## CUDA pattern used here

**Ensemble ODE integration over a virtual population**: one thread per patient,
full RK4 loop in registers, no inter-thread communication · shared
`__host__ __device__` model + RNG for exact CPU/GPU parity · double precision.

## Exercises

1. **More compartments.** Extend to liver/kidney/fat/muscle (flow-limited tissues)
   toward a real ~15-compartment PBPK; the parallel pattern is unchanged.
2. **Stiff solver.** Some PBPK systems are stiff; replace RK4 with an implicit
   (Rosenbrock/RODAS) solver and compare (the nvQSP approach).
3. **Multiple doses.** Add repeated dosing (a dosing schedule) and report steady-state Cmax/Ctrough.
4. **Sensitivity.** Vary one parameter's CV and measure its effect on AUC spread.
5. **Per-compound batch.** Loop over many compounds × the population (a 2-D launch).

## Limitations & honesty

- **3-compartment teaching reduction** of full ~15-compartment PBPK; fixed-step RK4
  (no stiff/adaptive handling); single oral dose; parameters are illustrative, not
  fitted to a drug.
- Log-normal sampling via a shared deterministic RNG (for exact CPU/GPU parity), not
  a validated population model.
