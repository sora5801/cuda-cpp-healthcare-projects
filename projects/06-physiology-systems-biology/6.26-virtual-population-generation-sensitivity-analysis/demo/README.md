# Demo — 6.26 Virtual Population Generation & Sensitivity Analysis

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/vpop_config.txt` — a synthetic study of 4096
   Saltelli base samples over k=4 uncertain PK parameters (24,576 model runs).
3. **Verify** two things against the CPU reference (`reference_cpu.cpp`): that the
   raw per-patient AUC array agrees to ~machine precision, **and** that the Sobol
   sensitivity indices computed from each array agree. Prints `PASS`/`FAIL`.
4. **Time** the GPU kernel (CUDA events) vs the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing, the numeric error, and the analytic
  cross-check (which vary run to run), so it is shown but never diffed.

## Expected result

```
6.26 -- Virtual Population Generation & Sensitivity Analysis
virtual population: 4096 Saltelli base samples, k=4 params, 24576 model evals
PK model: 1-compartment oral, dose=100 mg, AUC over 72 h (720 trapezoid steps)
parameter ranges (uniform priors):
  ka  in [0.500, 2.000]
  CL  in [3.000, 8.000]
  V   in [20.000, 50.000]
  F   in [0.600, 1.000]
population AUC: mean=15.6820  variance=25.8025  (mg.h/L)
Sobol sensitivity indices (fraction of AUC variance):
  param   S1(first-order)   ST(total-order)
  ka            0.0000           0.0000
  CL            0.7908           0.8018
  V            -0.0001           0.0000
  F             0.1949           0.2153
dominant parameter (largest S1): CL
RESULT: PASS (GPU matches CPU; raw AUC and Sobol indices within tol=1.0e-09)
```

## How to read it

The whole demo is a **self-checking science experiment**. The model's exposure
metric has a closed form, `AUC = F·Dose/CL`, which depends **only** on
bioavailability `F` and clearance `CL` — *not* on absorption rate `ka` or volume
`V`. So a correct global sensitivity analysis **must** attribute essentially all
of the AUC variance to `CL` and `F` and ~zero to `ka` and `V`:

- `S1(ka) ≈ 0`, `S1(V) ≈ 0` — these knobs do not move total exposure. ✔
- `S1(CL) = 0.79`, `S1(F) = 0.19` — `CL` is the dominant driver; `F` matters too. ✔
- `S1(CL)+S1(F) = 0.986 ≈ 1` — the two relevant parameters explain the variance. ✔

The stderr line `[science] ... -> CONSISTENT` confirms this analytic check, and
`RESULT: PASS` confirms the GPU reproduces the CPU reference to ~1e-14. On the
test machine the GPU ran the 24,576 model evaluations ~30× faster than the CPU.

> Parameters are **synthetic and illustrative** (a teaching-scale 1-compartment
> model), not fitted to any drug — a software demonstration, not a
> pharmacokinetic prediction, and not for any clinical or dosing use.
