# Demo — 1.1 Molecular Dynamics Engine

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed synthetic Lennard-Jones fluid
   (`data/sample/lj_sample.txt`): 27 atoms, 50 velocity-Verlet steps.
3. **Verify** the GPU trajectory against the CPU reference (`reference_cpu.cpp`)
   and print a clear `PASS`/`FAIL`.
4. **Time** the GPU kernels (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries timing and the run-varying CPU↔GPU numeric differences, so it
  is shown but never diffed.

## The teaching point: energy conservation

The headline result is that **total energy is conserved**. A correct, symplectic
velocity-Verlet integrator keeps `E_final ≈ E0` with only a tiny bounded drift —
here a *relative* drift around `2.5e-8`. That near-constant total energy is the
single best sign that an MD integrator is working, and it is what makes long
trajectories trustworthy. The negative total energy reflects atoms sitting in the
attractive Lennard-Jones well.

## Expected result

```
1.1 -- Molecular Dynamics Engine
Lennard-Jones fluid, velocity-Verlet (reduced units)
atoms = 27  box = 3.600  dt = 0.0040  steps = 50  rcut = 2.500
E0          = -103.417311
E_final     = -103.417308
max |dE|    = 2.534680e-06
rel drift   = 2.450924e-08
T_final     = 0.000286
pos_chksum  = 151.214071
RESULT: PASS (GPU matches CPU: dE<=1.0e-06, dchksum<=1.0e-04)
```

`E0`/`E_final` are the total energy at the start/end; `max |dE|` is the largest
energy excursion seen during the run; `pos_chksum` fingerprints the final atom
configuration (any CPU↔GPU trajectory divergence would change it). The CPU and GPU
match to ~`1e-13` here (see stderr); the small documented tolerances absorb the
GPU's fused-multiply-add / summation-order round-off (see `../THEORY.md §Numerics`).
