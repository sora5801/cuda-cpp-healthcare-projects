# 9.02 — Large-Scale Compartmental & Metapopulation Models

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Epidemiology%20%26%20Public%20Health-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 9: Epidemiology & Public Health · Catalog ID `9.02`
>
> _Educational only — not for clinical/forecasting use (see CLAUDE.md §8)._

## Summary

Run a whole **ensemble** of epidemic simulations at once: the same SEIR
compartmental ODE solved for thousands of parameter combinations (a sweep of
transmission rate β and recovery rate γ). Each parameter set is an independent
RK4 time-integration, so each GPU thread integrates one trajectory — the natural
pattern for **Monte Carlo / uncertainty quantification**. Eighth distinct GPU
pattern in the flagships: **parallel ensemble ODE integration**.

## What this computes & why the GPU helps

Compartmental models (SIR/SEIR) are systems of ODEs for the fractions of a
population that are Susceptible, Exposed, Infectious, Recovered. To quantify
uncertainty (which parameters, which outcomes?) you must solve the ODE for many
parameter samples — thousands to millions. These solves are independent, so the
GPU runs one per thread and reaches large speed-ups (the demo shows ~24× for 4096
members; the gap grows with ensemble size).

**The parallelized work** is the ensemble of RK4 integrations; each thread runs a
full time loop in registers and writes a summary (peak infection, attack rate).

## The algorithm in brief

- **SEIR ODE:** `dS=-βSI/N`, `dE=βSI/N-σE`, `dI=σE-γI`, `dR=γI` (R0 = β/γ).
- **RK4:** 4th-order Runge-Kutta time stepping (O(dt⁴) accurate, stable).
- **Ensemble:** integrate every (β, γ) on the sweep grid; collect peak infection,
  peak day, and attack rate per member.

See [THEORY.md](THEORY.md) for the model, RK4, and the metapopulation extension.

## Build

Requires **Visual Studio 2026** (v145) + **CUDA 13.3** ([docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/large-scale-compartmental-metapopulation-models.sln`.
2. **`Release|x64`** → **Build** → `build/x64/Release/large-scale-compartmental-metapopulation-models.exe`.

CLI: `msbuild build\large-scale-compartmental-metapopulation-models.sln /p:Configuration=Release /p:Platform=x64`

## Run the demo

```powershell
./demo/run_demo.ps1
```

Integrates the ensemble on CPU + GPU and verifies the per-member results match.

## Data

- **Sample (committed):** `data/sample/ensemble_params.txt` — a 64×64 β×γ sweep.
- **Realistic models:** MEmilio / EpiModel / Torchdiffeq — see
  `scripts/download_data.ps1` and [data/README.md](data/README.md).
- Bigger ensemble: `python scripts/make_synthetic.py --nb 200 --ng 200`.

## Expected output

`demo/expected_output.txt` holds the deterministic sample trajectories and
ensemble summary. The GPU (`src/kernels.cu`) and CPU (`src/reference_cpu.cpp`)
share the RK4 integrator (`src/seir.h`) and use **double precision**, so they
agree to ~machine precision (`worst diff ≈ 1e-15`).

## Code tour

1. [`src/main.cu`](src/main.cu) — load, CPU + GPU integrate, verify, print samples + summary.
2. [`src/seir.h`](src/seir.h) — **the SEIR ODE + RK4 step + per-member integrator** (host + device).
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU ensemble interface (one thread per member).
4. [`src/kernels.cu`](src/kernels.cu) — the ensemble kernel + host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the serial reference.

## Prior art & further reading

- **MEmilio** (<https://github.com/SciCompMod/memilio>) — high-performance C++/CUDA epidemic simulation.
- **EpiModel** (<https://github.com/EpiModel/EpiModel>) — network-based compartmental modelling.
- **Torchdiffeq** (<https://github.com/rtqichen/torchdiffeq>) — GPU ODE/neural-ODE solvers (`dopri5`).
- **PyGOM** (<https://github.com/ukhsa-collaboration/pygom>) — compartmental ODE modelling framework.

Study these for production modelling; reimplement the pattern didactically (CLAUDE.md §2).

## CUDA pattern used here

**Ensemble ODE integration**: one thread per parameter set, full RK4 loop in
registers, no inter-thread communication · shared `__host__ __device__` integrator
for exact CPU/GPU parity · double precision for accuracy over many steps.

## Exercises

1. **Random sampling.** Replace the grid sweep with Latin-hypercube or Sobol
   samples of (β, γ, σ) — proper Monte Carlo uncertainty quantification.
2. **Adaptive step size.** Implement RK45 (Dormand-Prince) with error control and
   compare cost/accuracy to fixed-step RK4.
3. **Save trajectories.** Write the full I(t) curve per member (a 2-D output) and
   plot the ensemble envelope.
4. **Age structure.** Expand each compartment into age bands with a contact
   matrix — more ODEs per member, same parallel pattern.
5. **Metapopulation coupling.** Couple many geographic patches by a mobility
   matrix; the per-step update becomes a batched sparse mat-vec (cuSPARSE).

## Limitations & honesty

- **Independent SEIR members** (no spatial coupling). The metapopulation case —
  many patches linked by mobility, a cuSPARSE batched SpMV per step — is described
  in THEORY and is Exercise 5.
- Fixed-step RK4 (no adaptive/stiff handling); a deterministic ODE (no demographic
  stochasticity). Parameter ranges are illustrative, not fitted.
