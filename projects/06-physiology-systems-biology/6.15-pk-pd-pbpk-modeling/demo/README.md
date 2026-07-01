# Demo — 6.15 PK/PD & PBPK Modeling

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/pkpd_params.txt` (4096 virtual patients, 48 h).
3. **Verify** that the GPU population matches the CPU reference (same coupled
   PK/PD RK4 in double precision) and print a clear `PASS`/`FAIL`.
4. **Report** a few sample patients (PK: Cmax/Tmax/AUC | PD: Rmax/Tresp/effect)
   and the population summary.

The program splits its output deliberately (PATTERNS.md §3):

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timings and the numeric verification error (which vary
  run to run), so it is shown but never diffed.

## Canonical output

See [`expected_output.txt`](expected_output.txt). Each patient's plasma curve peaks
after oral absorption (Tmax ≈ 1–2 h) and the population's **mean AUC ≈ dose/CL =
20** mg·h/L (the textbook PK identity), with patient-to-patient variability in the
AUC range. On the PD side the drug inhibits biomarker loss, so the biomarker rises
above its baseline **R0 = kin/kout = 50** and the mean peak fractional **effect**
is positive. `RESULT: PASS` means the GPU and CPU per-patient PK **and** PD metrics
agree to ~machine precision.

> Parameters are **illustrative** (a coupled 1-cpt-PK + indirect-response-PD
> teaching reduction of full PBPK/QSP), **not** fitted to any drug — a software
> demonstration, not a pharmacokinetic prediction, and not for clinical use.
