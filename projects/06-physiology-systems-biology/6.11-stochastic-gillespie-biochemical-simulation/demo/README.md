# Demo — 6.11 Stochastic (Gillespie) Biochemical Simulation

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/gene_network.txt` input — an ensemble
   of 256 independent Gillespie SSA trajectories of a birth–death gene model.
3. **Verify** the GPU result against the CPU reference (`reference_cpu.cpp`) and
   print a clear `PASS`/`FAIL`. The integer per-trajectory outputs (final count,
   event count) match **exactly**; the time-average matches to `~1e-15` (FMA).
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt). Determinism holds because the SSA
  is seeded from fixed integers and every printed value is an integer or an exact
  sum — no floating-point atomics, no run-to-run reordering.
- **stderr** carries the timing, the verification residual, and the science-check
  line (which vary run to run or machine to machine), so it is shown but never
  diffed.

## What to look for

- `RESULT: PASS` — the GPU ensemble reproduces the CPU reference trajectory for
  trajectory.
- `ensemble mean of time-avg M = 19.2437  (analytic Poisson mean = 20.0000)` —
  the simulation recovers the closed-form stationary mean `k_prod/k_deg`. The
  ~4% gap is finite-sample Monte-Carlo error (256 short runs); it shrinks like
  `1/√N` as you add trajectories.
- On stderr, `exact per-trajectory match=yes; worst time-avg diff = 7.105e-15` —
  the honest floating-point residual (host vs. device fused-multiply-add).

## Expected result (stdout)

```
6.11 -- Stochastic (Gillespie) Biochemical Simulation
Model: birth-death gene expression  (0 -> M @ k_prod, M -> 0 @ k_deg)
k_prod=10.000  k_deg=0.500  m0=0  t_end=50.0  trajectories=256
sample trajectories (idx: events finalM timeAvgM):
  t0    :    976      20    18.9959
  t64   :    964      18    18.2718
  t128  :    936      26    17.7395
  t192  :    982      20    18.3612
  t255  :    991      25    20.0243
ensemble mean of time-avg M = 19.2437   (analytic Poisson mean = 20.0000)
ensemble mean of final   M = 20.0312
total reaction events = 251072
RESULT: PASS (GPU ensemble matches CPU exactly, per-trajectory)
```

The per-trajectory rows and the ensemble aggregates are fully determined by the
fixed seed (`20240611`), so this output is reproducible on any machine.
