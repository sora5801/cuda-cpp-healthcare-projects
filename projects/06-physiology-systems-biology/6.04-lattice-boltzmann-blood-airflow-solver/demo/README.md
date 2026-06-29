# Demo — 6.04 Lattice-Boltzmann Blood/Airflow Solver

## What this demonstrates

`run_demo.ps1` (Windows) / `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/channel_params.txt` (a 16×24 channel, 6000 steps).
3. **Verify** that the GPU stencil produces the **same velocity field** as the
   CPU reference.
4. **Report** the across-channel velocity profile and the centerline maximum.

stdout (the deterministic velocity profile) is diffed against
[`expected_output.txt`](expected_output.txt); the timing line is on stderr only.

## Canonical output

See [`expected_output.txt`](expected_output.txt). The profile is a clean
**parabola** — zero at the no-slip walls, maximum at the centerline — i.e. the
LBM reproduces analytic **Poiseuille flow**. `RESULT: PASS` means the GPU and CPU
velocity fields agree (to ~machine precision here, since both use the same
double-precision per-node update).

> Geometry is a **simplified 2-D channel**, not a real vessel — a demonstration
> of the LBM/GPU pattern, not a validated hemodynamics simulation.
