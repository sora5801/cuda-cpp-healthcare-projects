# Data — 6.15 PK/PD & PBPK Modeling

## Committed sample (`sample/pkpd_params.txt`)

| Field | Value |
|---|---|
| Origin | **Synthetic** population configuration (`scripts/make_synthetic.py`) |
| License | Public domain (CC0) — it is synthetic |
| Size | < 1 KB (one line) |
| Setup | 4096 virtual patients, coupled 1-compartment oral PK + indirect-response PD, 48 h |

The "data" is the **population setup**, not a table of measurements. Each virtual
patient's individual PK/PD parameters are sampled deterministically from a seeded
RNG inside the program (`src/pkpd.h`), so the whole study is reproducible and the
CPU and GPU produce the identical population.

### File format (one whitespace-separated line)

```
dose  ka  CL  Vc  kin  kout  Imax  IC50  cv  dt  steps  n_patients  seed
```

| Field | Meaning (units) |
|---|---|
| `dose` | oral dose (mg), placed into the gut depot at t=0 |
| `ka` | median first-order absorption rate (1/h) |
| `CL` | median plasma clearance (L/h) |
| `Vc` | median central (plasma) volume of distribution (L) |
| `kin` | zero-order biomarker production rate (response-units/h) |
| `kout` | first-order biomarker loss rate (1/h); baseline `R0 = kin/kout` |
| `Imax` | maximum fractional inhibition of biomarker loss, in `[0,1]` |
| `IC50` | plasma concentration giving half-maximal inhibition (mg/L) |
| `cv` | log-normal between-subject variability applied to sampled PK/PD params |
| `dt`, `steps` | RK4 step (h) and step count (run = `steps·dt` h) |
| `n_patients` | virtual population size (one GPU thread each) |
| `seed` | base RNG seed |

Default: `100 1 5 30 10 0.2 0.9 2 0.25 0.05 960 4096 99`. Two sanity checks a
learner can verify against the program output:

- **PK:** with (near-)complete absorption, mean **AUC ≈ dose/CL = 20** mg·h/L.
- **PD:** biomarker baseline **R0 = kin/kout = 50** units; the drug inhibits its
  loss, so R rises above 50 and `effect = (Rmax − R0)/R0 > 0`.

## "Full dataset" / realistic PK/PD & PBPK

There is no file to download — the population is generated. For **real** clinical
PK data and **validated** models (some require registration; the download scripts
link and instruct, they never scrape):

- **PhysioNet / MIMIC** (<https://physionet.org>) — clinical time series;
  **credentialed** access (register; do not bypass).
- **FDA FAERS** (<https://www.fda.gov/drugs/fda-adverse-event-reporting-system-faers>)
  — public adverse-event reports.
- **OSP PBPK Model Library**
  (<https://github.com/Open-Systems-Pharmacology/OSP-PBPK-Model-Library>) —
  whole-body PBPK models (~15 physiological compartments).
- **DDMoRe model repository** (<https://ddmore.eu/models-tools>) — curated
  pharmacometric (PK/PD/NLME) models.

Bigger synthetic population: `python scripts/make_synthetic.py --patients 100000`.

## Provenance & honesty

The configuration is **synthetic**, and the coupled 1-compartment-PK +
indirect-response-PD model is a **teaching reduction** of full PBPK/QSP; the
parameters are illustrative, **not fitted to any drug**. Outputs are a software
demonstration, **not** a pharmacokinetic prediction, and **not for any
clinical/dosing use** (CLAUDE.md §8).
