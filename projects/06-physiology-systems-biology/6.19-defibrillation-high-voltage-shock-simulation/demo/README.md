# Demo — 6.19 Defibrillation & High-Voltage Shock Simulation

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/defib_sweep.txt` input.
3. **Verify** the GPU result against the CPU reference (`reference_cpu.cpp`) and
   print a clear `PASS`/`FAIL`.
4. **Time** the sweep (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program sweeps a ladder of **defibrillation shock amplitudes** applied to a
1-D cardiac cable that starts with an ongoing travelling wave (a stand-in for
fibrillation). For each amplitude it reports the **residual electrical activity**
left after the shock: high means the shock *failed* (the wave survived), ~0 means
it *defibrillated* (the tissue was reset to rest). The smallest successful
amplitude is the **defibrillation threshold (DFT)**.

The output is split deliberately (PATTERNS.md §3):

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric GPU-vs-CPU error (which vary run
  to run), so it is shown but never diffed.

## Expected result

```
6.19 -- Defibrillation & High-Voltage Shock Simulation
1-D monodomain cable (FitzHugh-Nagumo), monophasic shock
cable: 100 cells, 2000 steps, dt=0.1000 dx=1.000 D=0.600
shock window: steps [800,810), success if residual < 0.0100
shock amplitude sweep (residual activity after shock):
  amp=0.000  residual=0.051915  failed
  amp=0.050  residual=0.052210  failed
  amp=0.100  residual=0.051709  failed
  amp=0.150  residual=0.000000  DEFIBRILLATED
  amp=0.200  residual=0.000000  DEFIBRILLATED
  amp=0.250  residual=0.000000  DEFIBRILLATED
  amp=0.300  residual=0.000000  DEFIBRILLATED
  amp=0.400  residual=0.000000  DEFIBRILLATED
  amp=0.600  residual=0.000000  DEFIBRILLATED
  amp=1.000  residual=0.000000  DEFIBRILLATED
DFT: amplitude 0.150 (index 3) -- weakest shock that terminated activity
RESULT: PASS (GPU matches CPU within tol=1.0e-06)
```

## How to read it

- The three weakest shocks (0.00–0.10) leave a residual travelling wave (~0.052):
  they are **too weak to defibrillate**.
- From amplitude **0.15** upward the residual collapses to 0 — the shock forces
  the whole cable past threshold and it recovers to rest. There is a clean,
  monotone **all-or-nothing threshold**, exactly the DFT concept.
- The GPU and CPU residuals agree to ~1e-17 (essentially machine precision)
  because both call the identical shared physics in `src/defib.h`.

The `stderr` timing line is illustrative only: this sweep (10 amplitudes) is tiny
and launch/copy-dominated, so the CPU can look faster — the GPU's advantage grows
with the number of amplitudes and the cable size (see `THEORY.md` §GPU mapping).
