# Data — 13.02 PBPK at Scale

## Committed sample (`sample/pbpk_params.txt`)

| Field | Value |
|---|---|
| Origin | **Synthetic** population configuration (`scripts/make_synthetic.py`) |
| License | Public domain (CC0) |
| Setup | 4096 virtual patients, 3-compartment oral PK, 48 h simulation |

The "data" is the **population setup**; each patient's parameters are sampled
deterministically from a seeded RNG (pbpk.h), so the study is reproducible.

### File format (one line)

```
dose  ka  CL  Vc  Vp  Q  cv  dt  steps  n_patients  seed
```

| Field | Meaning |
|---|---|
| `dose` | oral dose (mg), into the gut depot |
| `ka` | median first-order absorption rate (1/h) |
| `CL` | median clearance (L/h) |
| `Vc`, `Vp` | median central / peripheral volumes (L) |
| `Q` | median inter-compartment flow (L/h) |
| `cv` | log-normal variability applied to each sampled parameter |
| `dt`, `steps` | RK4 step (h) and step count (run = `steps·dt` h) |
| `n_patients` | virtual population size |
| `seed` | base RNG seed |

Default: `100 1 5 30 40 7 0.3 0.05 960 4096 99`. Sanity check: with full
absorption, mean **AUC ≈ dose/CL = 20** mg·h/L.

## "Full dataset" / realistic models

Real PBPK uses ~15 physiological compartments (liver, kidney, lung, fat, muscle,
gut, ...) with literature tissue volumes/blood flows and compound-specific
partition coefficients and metabolism:

- **PK-Sim** (<https://github.com/Open-Systems-Pharmacology/PK-Sim>) — whole-body PBPK.
- **nvQSP** (<https://github.com/NVIDIA-Digital-Bio/nvQSP>) — NVIDIA GPU QSP/PBPK ODE solvers.
- ICRP/Open Systems Pharmacology physiology databases for parameter values.

Bigger population: `python scripts/make_synthetic.py --patients 100000`.

## Provenance & honesty

The configuration is **synthetic** and the 3-compartment model is a teaching
reduction of full PBPK; parameters are illustrative, not fitted to any drug.
Outputs are a software demonstration, not a pharmacokinetic prediction, and are
not for any clinical/dosing use.
