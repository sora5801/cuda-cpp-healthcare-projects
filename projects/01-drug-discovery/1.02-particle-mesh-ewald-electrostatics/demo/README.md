# Demo — 1.2 Particle-Mesh Ewald Electrostatics

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/charges_sample.txt` (a synthetic
   64-ion NaCl-like crystal in a periodic box).
3. **Compute the reciprocal-space PME energy** three ways and **verify** them:
   - GPU smooth-PME (atomic B-spline charge spreading → cuFFT → convolve),
   - the same SPME pipeline on the **CPU** (the GPU's exact twin), and
   - the **direct Ewald** reciprocal sum over k-vectors (the gold standard).
4. Assemble the **full Ewald energy** (real + reciprocal − self) and show it is
   **invariant to the Ewald splitting parameter β** — the method's key physics
   self-consistency check.
5. **Time** the GPU (cuFFT) vs the CPU (naive DFT) — a *teaching artifact*, not a
   benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries timing and run-varying numbers, so it is shown but not diffed.

## What the numbers mean

- `E_recip (GPU SPME)` vs `(CPU SPME)` agree to ~`6e-8` relative — the only
  difference is **FP32 cuFFT vs FP64 host DFT** (the charge grid itself is
  bit-identical thanks to fixed-point spreading; see THEORY §verify).
- `E_recip (CPU SPME)` vs `(direct Ewald)` agree to ~`7e-12` — SPME reproduces the
  exact reciprocal sum at this grid/order (the **science** check).
- `E_total` ≈ `−27.96` reduced units, and the `β`-invariance check confirms the
  real/reciprocal/self decomposition is correct (a different β gives the same
  total to ~`3e-5`).

## Expected result

```
1.2 -- Particle-Mesh Ewald Electrostatics
system: 64 charges in a cubic box of side 8.0000 (reduced units)
PME params: grid 16x16x16, B-spline order 4, beta 0.750000, rcut 4.000000
E_recip (GPU SPME) = 2.02433442
E_recip (CPU SPME) = 2.02433454
E_recip (direct Ewald) = 2.02433454
E_real = -2.90499951   E_self = 27.08110001
E_total (real + recip - self) = -27.96176498
CHECK GPU==CPU SPME      : PASS (rel 6.15e-08 <= 1e-04)
CHECK SPME~=direct Ewald : PASS (rel 7.46e-12 <= 5e-03)
CHECK total invariant to beta : PASS (rel 2.62e-05 <= 2e-02)
RESULT: PASS
```

> The exact low-order digits of the GPU SPME energy can differ on a different GPU
> (FP32 cuFFT is hardware-specific); the verification tolerances absorb that, and
> `expected_output.txt` was captured on the development machine (RTX 2080, sm_75).
