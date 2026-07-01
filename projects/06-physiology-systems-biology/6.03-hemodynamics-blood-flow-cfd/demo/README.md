# Demo — 6.3 Hemodynamics / Blood-Flow CFD

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/channel_params.txt` (a 32×17 synthetic channel,
   40000 time steps, 40 Jacobi pressure iterations per step).
3. **Verify** that the GPU fractional-step solver produces the **same velocity
   field** as the CPU reference (`reference_cpu.cpp`) — `RESULT: PASS`.
4. **Report** the across-channel velocity profile, the centreline maximum, and
   the **wall shear stress (WSS)** at the wall.
5. **Time** the kernel loop (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately (PATTERNS.md §3):

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing, the science-level convergence error, and the
  CPU-vs-GPU difference (which vary run to run), so it is shown but never diffed.

## Canonical output

See [`expected_output.txt`](expected_output.txt). The velocity profile is a clean
**parabola** — zero at the no-slip walls, maximum at the centreline — i.e. the
solver reproduces analytic **Poiseuille flow**. The stderr line reports
`|u_max - analytic|/analytic ≈ 4.8%`: the simulation has converged to within ~5%
of the exact steady parabola (run more steps to close the gap — see the exercise
in the project README). `RESULT: PASS` means the GPU and CPU velocity fields agree
to a tight tolerance (here they are bit-identical).

## Honesty note

This is a **reduced-scope teaching version** (CLAUDE.md §13): a 2-D structured-grid
incompressible Navier-Stokes solver on a **simplified straight channel**, not a
real 3-D patient-specific vessel with compliant walls and fluid-structure
interaction. It demonstrates the CFD/GPU *pattern* (a fractional-step stencil with
a Jacobi pressure solve), not a validated hemodynamics simulation. The input is
**synthetic** — no patient data. See THEORY.md "Where this sits in the real world".
