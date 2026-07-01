# Demo — 6.24 Reaction-Diffusion Morphogenesis (Turing Patterns)

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/turing_params.txt` — a 64×64 grid,
   3000 timesteps of the Gierer–Meinhardt activator–inhibitor model.
3. **Verify** the GPU result against the CPU reference (`reference_cpu.cpp`,
   running the identical per-cell physics from `src/turing.h`) and print a clear
   `PASS`/`FAIL`.
4. **Check the science**: it also computes the analytic **Turing dispersion
   relation** and reports whether the parameters form a pattern and at what
   wavelength — an independent test of the simulation.
5. **Time** the GPU stepping loop (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing (which varies run to run), so it is shown but never
  diffed.

## What you are looking at

- `pattern:` line — summary statistics of the final **activator** field. `contrast`
  (`max − min`) far from zero means a real pattern formed (here `≈ 7.13`, on a field
  whose uniform steady state is `a* = 1.4`). `peak cells (a>mean)` counts the
  "on" regions — a proxy for the number of spots/stripes.
- `linear stability:` line — `Turing regime=YES` confirms the parameters sit in the
  pattern-forming region; `k*` and the `predicted wavelength` come from the 2×2
  eigenvalue analysis in `main.cu::turing_growth_rate`, *not* from the simulation,
  so their agreement with the observed pattern validates the physics.
- `RESULT: PASS` — the GPU and CPU final fields agree to within `1e-6`.

## Expected result

```
6.24 -- Reaction-Diffusion Morphogenesis (Turing Patterns)
Gierer-Meinhardt: 64x64 grid, 3000 steps, Da=0.0200 Dh=0.5000 (Dh/Da=25.0) rho=0.050 mu_a=0.100 mu_h=0.140
pattern: mean a=0.624556, min a=0.003645, max a=7.132460, contrast=7.128815, peak cells (a>mean)=907 of 4096
linear stability: Turing regime=YES, max growth=0.040278 at k*=1.1436 (predicted wavelength=5.49 cells)
a along center row (8 samples): 0.3352 0.0146 0.1802 0.0347 1.1905 1.1998 0.1173 0.0649
RESULT: PASS (GPU field matches CPU within tol=1.0e-06)
```

The exact numbers are reproducible because the initial noise is a deterministic
hash of `(x, y, noise_seed)` (not `rand()`), so the pattern is identical on every
machine and on both the CPU and GPU paths.
