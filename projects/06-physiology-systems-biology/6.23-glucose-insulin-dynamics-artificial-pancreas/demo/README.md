# Demo — 6.23 Glucose-Insulin Dynamics & Artificial Pancreas

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/cohort_params.txt` — **1024 virtual T1D patients**
   (a 32×32 sweep of insulin sensitivity `SI` and glucose effectiveness `SG`),
   each simulated for an 8-hour closed-loop scenario: a 50 g meal at t = 30 min
   with a PID controller dosing insulin every 5 minutes.
3. **Verify** that the GPU cohort matches the CPU reference (same RK4 + PID) and
   print a clear `PASS`/`FAIL`.
4. **Report** a few sample patients (min/max/mean glucose, time-in-range,
   hypoglycemia fraction, total insulin) and a cohort summary.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## Canonical output

See [`expected_output.txt`](expected_output.txt). The results are textbook glucose
physiology: **more insulin-sensitive patients** (larger `SI`) reach lower peak
glucose, higher time-in-range, and need **less** insulin — but their glucose nadir
drifts toward the 70 mg/dL hypoglycemia boundary, which is exactly the safety
tension a real artificial pancreas must manage. On this sample every patient stays
above 70 mg/dL (no hypoglycemia) and the mean time-in-range is ≈ 91%.

`RESULT: PASS` means the GPU cohort matches the CPU reference to ~machine precision
(the shared double-precision RK4 + PID core in `src/bergman.h`; worst per-patient
diff ≈ 1e-13). On the sample the GPU runs the 1024 patients several× faster than
the CPU — a gap that grows with cohort size (RL / uncertainty studies run 10³–10⁶
patients across many episodes).

> Parameters are **synthetic and illustrative**, not fitted to any patient — a
> software demonstration of ensemble closed-loop ODE simulation, **not** a medical
> device and **not** for any clinical decision.
