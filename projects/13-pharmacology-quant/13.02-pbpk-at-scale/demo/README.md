# Demo — 13.02 PBPK at Scale

## What this demonstrates

`run_demo.ps1` (Windows) / `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/pbpk_params.txt` (4096 virtual patients, 48 h).
3. **Verify** that the GPU population matches the CPU reference (same RK4).
4. **Report** a few sample patients (Cmax, Tmax, AUC) and the population summary.

stdout (the summary) is deterministic and diffed against
[`expected_output.txt`](expected_output.txt); the timing line is on stderr only.

## Canonical output

See [`expected_output.txt`](expected_output.txt). Each patient's plasma curve peaks
after oral absorption (Tmax ≈ 1–2 h) and the population's **mean AUC ≈ 19 ≈
dose/CL = 20** (the textbook PK identity), with patient-to-patient variability
(AUC range ≈ 5–42). `RESULT: PASS` means the GPU and CPU per-patient metrics agree
to ~machine precision (double-precision RK4). The GPU integrates the 4096 patients
~19× faster than the CPU; real QSP/PBPK studies run 10⁴–10⁶ ODE solves.

> Parameters are **illustrative** (a 3-compartment teaching reduction), not fitted
> to any drug — a software demonstration, not a pharmacokinetic prediction.
