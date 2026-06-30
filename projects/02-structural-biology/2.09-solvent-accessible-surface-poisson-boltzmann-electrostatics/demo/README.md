# Demo — 2.9 Solvent-Accessible Surface & Poisson-Boltzmann Electrostatics

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/molecule.pqr` (11-atom synthetic dipole; 48³ grid,
   600 red-black Gauss–Seidel sweeps).
3. **Verify** that the GPU potential field matches the CPU reference field
   (`reference_cpu.cpp`) — both run the identical red-black relaxation — and print
   `PASS`/`FAIL`.
4. **Report** the electrostatics summary (extreme/centre potentials), the SASA,
   and a potential profile along the central x-line.
5. **Time** the GPU solve (CUDA events) vs the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the verification error (which vary run to
  run), so it is shown but never diffed.

## Canonical output

See [`expected_output.txt`](expected_output.txt):

```
2.9 -- Solvent-Accessible Surface & Poisson-Boltzmann Electrostatics
grid: 48x48x48 cells, h=0.60 A, eps_in=2.0 eps_out=80.0 kappa^2=0.1000, sweeps=600
atoms: 11  | SASA (probe=1.4 A) = 408.84 A^2
potential (kT/e): min=-0.367998  max=0.367998  center=-0.005102  sum|phi|=163.6273
phi along center x-line (8 samples): 0.0000 0.0035 0.0340 0.0486 -0.0295 -0.0516 -0.0047 0.0000
RESULT: PASS (GPU field matches CPU within tol=1.0e-09)
```

What to notice:

- **`min = −max`** (−0.368, +0.368): the potential is **antisymmetric**, exactly
  as the symmetric +/− dipole demands. `center ≈ 0` is the dipole midpoint. This
  is a physical sanity check, not just CPU==GPU.
- **`RESULT: PASS`** means the GPU and CPU fields agree to ~`5.6e-17` (machine
  precision; tolerance 1e-9). On stderr you will see the GPU run the 600-sweep
  solve several times faster than the serial CPU sweep — the edge grows with grid
  size; tiny grids are launch-bound.

> The molecule is an **abstract synthetic dipole** in reduced units — a
> demonstration of the finite-difference red-black-stencil PB solve, **not** a
> calibrated pKa/binding/zeta calculation and not for any clinical use.
