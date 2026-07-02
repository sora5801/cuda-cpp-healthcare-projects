# Data — 6.26 Virtual Population Generation & Sensitivity Analysis

## Committed sample (`sample/vpop_config.txt`)

| Field | Value |
|---|---|
| File | `sample/vpop_config.txt` |
| Origin | **Synthetic** study configuration (`scripts/make_synthetic.py`) |
| License | Public domain (CC0) — it is synthetic |
| Size | < 1 KB |
| Setup | 4096 Saltelli base samples, k=4 PK params, 1-compartment oral model |

The committed "data" is a **study configuration**, not a patient table. The
virtual patients are generated *deterministically inside the program* from a
Halton quasi-random sequence (`src/vpop.h`), so the whole study reproduces from
this one small file — no per-patient rows to ship, and `demo/run_demo` runs
**offline with zero downloads** (CLAUDE.md §8).

### File format (whitespace-separated; the loader skips newlines)

```
dose                # oral dose (mg)
ka_lo  ka_hi        # first-order absorption rate range (1/h)
CL_lo  CL_hi        # clearance range (L/h)
V_lo   V_hi         # central distribution volume range (L)
F_lo   F_hi         # oral bioavailability range (unitless fraction in (0,1])
t_end  steps        # AUC integration horizon (h) and trapezoid step count
N      seed         # Saltelli base sample size, base RNG seed (reserved)
```

Default sample:

```
100
0.5 2
3 8
20 50
0.6 1
72 720
4096 99
```

| Field | Meaning |
|---|---|
| `dose` | administered oral dose (mg), a fixed constant |
| `ka` range | absorption rate: how fast drug enters plasma |
| `CL` range | clearance: how fast the body removes drug (drives AUC) |
| `V` range | central volume of distribution |
| `F` range | oral bioavailability: fraction of dose reaching circulation |
| `t_end`, `steps` | horizon and grid for the trapezoid AUC integral |
| `N` | Saltelli base sample size → `N·(k+2) = N·6` total model evaluations |
| `seed` | reserved (the Halton sequence itself is deterministic) |

**Sanity check.** The exposure metric has a closed form `AUC = F·Dose/CL`, so it
depends only on `F` and `CL`. A correct Sobol analysis therefore attributes
~all AUC variance to `CL` and `F` and ~0 to `ka` and `V` — the built-in teaching
check the demo prints.

Bigger study (no download): `python scripts/make_synthetic.py --N 16384`.

## "Full dataset" / realistic virtual populations

A production virtual-population + sensitivity workflow replaces our uniform
priors and toy model with measured physiology and full PBPK:

- **NHANES** anthropometric/physiological data — <https://www.cdc.gov/nchs/nhanes/>
  (body weight, organ sizes, demographics for realistic parameter distributions).
- **WHO growth reference data** — <https://www.who.int/tools/growth-reference-data-for-5to19-years>.
- **OSP PBPK Model Library** — <https://github.com/Open-Systems-Pharmacology/OSP-PBPK-Model-Library>
  (whole-body PBPK models; PK-Sim's virtual-population module).
- **FDA drug-label PK data** — <https://www.fda.gov/drugs> (clearance, volume,
  bioavailability priors for real compounds).
- **SALib** — <https://github.com/SALib/SALib> (reference Morris/Sobol/FAST
  implementations to cross-check our indices).

These are external, separately-licensed resources. `scripts/download_data.*`
prints these pointers and does **not** attempt to bypass any registration.

## Provenance & honesty

The configuration is **synthetic** and the 1-compartment oral PK model is a
teaching reduction; the parameter ranges are illustrative, not fitted to any
drug. Outputs are a software demonstration of the Saltelli/Sobol algorithm on a
GPU — **not** a pharmacokinetic prediction, and **not** for any clinical or
dosing decision.
