# Demo — 2.27 Polarizable Water Model GPU Dynamics

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/water_cluster.txt` (an isolated
   polarizable probe + two water-like molecules).
3. **Solve** the self-consistent induced dipoles on the GPU (Jacobi SCF, one
   thread per site) and on the CPU reference, and **verify** they agree.
4. **Time** the kernels (CUDA events) vs. the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately (PATTERNS.md §3):

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries timing and the verification residuals (which vary run to
  run), so it is shown but never diffed.

## Canonical output

See [`expected_output.txt`](expected_output.txt). What to read in it:

- `converged in 10 sweeps` — the mutual-polarization fixed point is reached in 10
  Jacobi sweeps (the two waters' dipoles keep nudging each other until stable).
- The per-site `|mu|` table — the two oxygens (sites 1, 4) carry induced dipoles;
  the bare hydrogens (`alpha = 0`) carry none. The two oxygens get *different*
  dipoles because the cluster geometry is asymmetric.
- `polarization energy U_pol = ... = -196.527992 kcal/mol` — the induction energy
  the polarizable model adds on top of fixed-charge electrostatics.
- `probe check: |mu0| = 0.072198272  analytic alpha*Eext = 0.072200000` — the
  isolated probe recovers the **analytic** `µ = αE` to ~2×10⁻⁶ (the residual is
  the cluster's field at 50 Å — physics, not a bug). This validates the *science*,
  not just CPU==GPU agreement.
- `RESULT: PASS` — the GPU dipoles and energy match the CPU reference to ≤1e-9
  (they run identical double-precision arithmetic and reduce in fixed point).

The stderr line shows the worst CPU-vs-GPU dipole difference (~2×10⁻¹⁶, i.e.
round-off) and that both solvers take the same number of sweeps.

> The cluster, charges, and polarizabilities are a **simplified teaching model**,
> not a fitted force field — a software demonstration of the self-consistent
> induced-dipole solve, not a water-model validation.
