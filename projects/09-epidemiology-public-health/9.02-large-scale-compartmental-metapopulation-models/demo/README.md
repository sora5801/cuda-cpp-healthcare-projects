# Demo — 9.02 Large-Scale Compartmental & Metapopulation Models

## What this demonstrates

`run_demo.ps1` (Windows) / `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/ensemble_params.txt` (4096 SEIR members = a 64×64
   sweep of transmission rate β and recovery rate γ).
3. **Verify** that the GPU ensemble matches the CPU reference (same RK4).
4. **Report** a few sample trajectories (peak infection, peak day, attack rate)
   and an ensemble summary.

stdout (the deterministic summary) is diffed against
[`expected_output.txt`](expected_output.txt); the timing line is on stderr only.

## Canonical output

See [`expected_output.txt`](expected_output.txt). The results are textbook
epidemiology: higher R0 = β/γ gives an earlier, taller infection peak and a larger
attack rate; members with R0 < 1 fizzle out. `RESULT: PASS` means the GPU and CPU
agree to ~machine precision (double-precision RK4). On the sample the GPU runs the
4096 trajectories ~24× faster than the CPU — a gap that grows with ensemble size
(real uncertainty-quantification runs 10⁴–10⁶ members).

> Parameter ranges are **illustrative**, not fitted to any disease — a software
> demonstration of ensemble ODE integration, not an epidemic forecast.
