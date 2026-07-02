# Demo — 6.22 Bone Remodeling Simulation

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed synthetic sample `data/sample/bone_params.txt`.
3. **Verify** the GPU remodeled-density field against the CPU reference
   (`reference_cpu.cpp`) and print a clear `PASS`/`FAIL`.
4. **Time** all GPU kernels (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately (PATTERNS.md §3):

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## Expected result

```
6.22 -- Bone Remodeling Simulation
[reduced-scope teaching model: mechanostat remodeling on a 2-D voxel grid]
grid 24x16, 60 remodel steps, 80 Jacobi sweeps/step
localized load 4.00 on top-edge columns [10,13]
mechanostat: setpoint k=0.550, lazy zone +/-0.200, rate=0.050, rho in [0.050,1]
total bone mass (sum rho) = 33.614792
mechanostat state: resorbing=366 homeostatic=8 forming=10 (of 384 voxels)
per-column bone mass profile (x=0..23):
 0.8000 0.8000 0.8000 0.8000 0.8000 0.8000 0.8000 0.8787 1.5097 2.3313 3.1786 3.3091 3.3091 3.1786 2.3313 1.5097 0.8787 0.8000 0.8000 0.8000 0.8000 0.8000 0.8000 0.8000
RESULT: PASS (GPU density matches CPU within tol=1.0e-09)
```

## How to read it (the science you should see)

- The **per-column bone mass profile** is the headline. Load is pushed in only on
  the center of the top edge (columns 10–13), and the profile **peaks right under
  that footprint** (~3.31 at the load center) while the lightly-loaded flanks
  **resorb down to the floor** (`0.8000` = 16 rows × the `rho_min` = 0.05 floor).
  That peak is a load-aligned **trabecular strut** — a qualitative illustration of
  **Wolff's law** ("bone adapts to the loads it carries").
- The **mechanostat state histogram** shows most voxels have driven their signal
  below the lazy zone (resorbing) because so much material was originally present;
  the strut voxels sit near or above homeostasis. Because these are *integer*
  counts, this line is exactly reproducible.
- The **total bone mass** dropped from the initial `24*16*0.5 = 192` toward `33.6`:
  the specimen shed material it was not using and kept a lean, load-bearing strut.
- **`RESULT: PASS`** means the GPU density field matched the CPU reference to
  within `1e-9` (the observed max difference is ~`1.1e-16`, i.e. essentially
  bit-identical — the shared `__host__ __device__` physics does its job).

## Notes

- On this tiny 24×16 grid the GPU is *slower* than the CPU (the many small kernel
  launches are launch-bound). That is honest and expected; the GPU's advantage
  grows with grid size (a real µCT volume is 10⁸ voxels). See THEORY.md §7.
- This is a **reduced-scope teaching model** (CLAUDE.md §13): it keeps the
  remodeling *biology* but replaces the production finite-element stress solve
  with a cheap density-weighted diffusion proxy. THEORY.md "real world" says how
  production differs.
