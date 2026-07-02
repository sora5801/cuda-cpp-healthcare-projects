# Demo — 6.10 Systems-Biology ODE/SDE Network Solver

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/ensemble_params.txt` — a synthetic
   36-member (`6 alpha × 6 n`) sweep of the repressilator gene circuit.
3. **Integrate** every member on the GPU (one thread per member, full RK4 loop)
   and on a CPU reference, and **verify** they agree: continuous observables
   within `1e-9`, and the integer oscillation flag exactly. Prints `PASS`/`FAIL`.
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic (and identical across Debug/Release
  builds) and is diffed against [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric verification detail (which vary
  run to run), so it is shown but never diffed.

## Expected result

```
6.10 -- Systems-Biology ODE/SDE Network Solver
repressilator ensemble: 36 members (6 alpha x 6 n), 4000 steps @ dt=0.050 (T=200), beta=5.0 alpha0=1.0
sample members (alpha n -> p2_final p2_min p2_max crossings osc):
  m0  :   10.0 1.00 ->    3.3166    3.3166    3.3166   0 0
  m9  :   60.0 2.20 ->    5.6226    2.7848    5.7201  17 1
  m18 :  160.0 1.00 ->   12.6886   12.6886   12.6886   0 0
  m27 :  210.0 2.20 ->    2.2663    2.2477   22.5924  14 1
  m35 :  260.0 3.00 ->    2.6560    1.3408   57.4447  10 1
ensemble: 17/36 members sustain oscillations
RESULT: PASS (GPU ensemble matches CPU within tol=1.0e-09)
```

## How to read it

Each printed member shows `alpha n -> p2_final p2_min p2_max crossings osc`,
tracking the readout protein `p2`:

- **m0, m18** have Hill coefficient `n = 1.0` (weak cooperativity). The circuit
  settles to a **steady state**: `p2_min == p2_max == p2_final`, 0 crossings,
  `osc = 0`.
- **m9, m27, m35** have higher `n` and/or `alpha`. The negative-feedback ring
  destabilises into a sustained **oscillation**: a wide `[min,max]` band, many
  level crossings, `osc = 1`.

The headline `17/36 members sustain oscillations` recovers the repressilator's
known behaviour — oscillation switches on as cooperativity and transcription
strength increase (see [../THEORY.md](../THEORY.md) §2, §6). Timing appears on
stderr; on tiny 36-member inputs the GPU is launch-bound and can be *slower* than
the CPU — the GPU's edge grows with ensemble size (Exercise 1).
