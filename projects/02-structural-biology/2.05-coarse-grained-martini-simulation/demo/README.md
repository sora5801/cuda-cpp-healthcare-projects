# Demo — 2.5 Coarse-Grained / MARTINI Simulation

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/cg_system.txt` (16 coarse-grained beads: 8 apolar
   "C" + 8 polar "P", in a 6 nm periodic box, for 200 velocity-Verlet steps).
3. **Verify** that the GPU MD trajectory matches the CPU reference
   (`reference_cpu.cpp`) on the final bead positions, and print `PASS`/`FAIL`.
4. **Report** the system's total energy, the C/P demixing order parameter, and a
   few sampled final positions.
5. **Time** the kernel loop (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic (positions printed to 4 decimals, so
  the ~`1e-11` CPU/GPU drift never shows) and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the verification error (which vary run to
  run), so it is shown but never diffed.

## Canonical output

See [`expected_output.txt`](expected_output.txt). The key lines:

- `total energy: initial = -108.6567   final = -108.8853` — the energy is nearly
  conserved and drops slightly as the beads settle into a lower-energy demixed
  arrangement. A stable energy is the headline sanity check on the integrator.
- `C/P centroid separation: ... 2.4000 nm ... 2.3936 nm` — the apolar C and polar
  P groups stay separated (the "like-likes-like" `eps` matrix keeps oil and water
  demixed); the small change is the clusters tightening.
- `RESULT: PASS (GPU trajectory matches CPU within tol=1.0e-06)` — the GPU and CPU
  agree to ~`1e-11` (reported on stderr), far inside the `1e-6` tolerance.

> **Numerical note:** the CPU and GPU run the *same* pair sum in the *same* index
> order (shared `martini.h`), so they agree to near machine precision; the residual
> ~`1e-11` is the GPU's fused-multiply-add contraction differing from the host
> compiler's — a real lesson in GPU reproducibility (see THEORY §5–6).

> The system is a **synthetic two-type bead box**, not a real lipid membrane — a
> demonstration of the non-bonded CG-MD GPU pattern, not a validated molecular
> model, and **not for any clinical use**.
