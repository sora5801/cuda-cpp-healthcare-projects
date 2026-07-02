# Demo — 6.27 Parameter Estimation & Data Assimilation for Physiological Models

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/enkf_config.txt` (a synthetic twin
   experiment: 256 ensemble members, 40 observation windows).
3. **Verify** the GPU-forecast filter against the fully-CPU filter
   (`reference_cpu.cpp`) member-for-member, and print a clear `PASS`/`FAIL`.
4. **Time** the GPU forecast (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## Expected result

```
6.27 -- Parameter Estimation & Data Assimilation for Physiological Models
two-element Windkessel; EnKF joint state-parameter estimation
ensemble=256  windows=40  window=0.080s (16 x dt=0.005s)  obs_noise=1.0 mmHg
true    : R=1.0000 mmHg*s/mL   C=1.5000 mL/mmHg
prior   : R=1.4000 (40.0% off)   C=1.0000 (33.3% off)
estimate: R=0.9986 (0.14% err)   C=1.4675 (2.17% err)
posterior spread: R_std=0.0030   C_std=0.0173
final ensemble-mean pressure RMSE vs obs = 1.7036 mmHg
RESULT: PASS (GPU EnKF matches CPU within tol=1.0e-06)
```

## How to read it

- The filter starts from a **deliberately-wrong prior** (R 40% off, C 33% off) and
  the ensemble Kalman updates pull it toward the truth: R lands within ~0.1% and C
  within ~2%. That recovery is the *method* working — see THEORY §6.
- `RESULT: PASS` means the GPU-forecast filter and the all-CPU filter produced the
  same final ensemble to ~round-off (worst per-member diff `≈ 1e-14`, on stderr).
- C is intrinsically harder to identify than R from end-of-window pressures alone,
  so its residual error is larger — an honest observability lesson, not a bug.

> The sample is **synthetic** (a twin experiment), not a patient measurement — a
> software demonstration of ensemble data assimilation, not a clinical tool.
