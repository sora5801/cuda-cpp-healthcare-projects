# Demo — 4.17 Real-Time Intraoperative / Image-Guided Surgery

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/surface_pair.txt` — a tiny synthetic
   pair of 3-D surfaces (a misaligned "pre-op" cloud `P` and a "intra-op" cloud
   `Q`).
3. **Register** them with GPU **Iterative Closest Point (ICP)** and, in parallel,
   the CPU reference; **verify** the GPU-recovered rigid transform matches the
   CPU one and print a clear `PASS`/`FAIL`.
4. **Time** the correspondence/reduction kernels (CUDA events) versus the CPU
   baseline — a *teaching artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the verify diagnostics (which vary run to
  run), so it is shown but never diffed.

## What to look for

- **The RMS convergence curve.** It starts at ~3.2 mm (after the coarse centroid
  pre-alignment) and drops to **~0.24 mm** — the surface-noise floor — in a single
  ICP iteration, then holds flat. That flat plateau is ICP having converged.
- **The recovered transform `[R | t]`.** Its rotation is the inverse (transpose)
  of the ~12° ground-truth rotation the sample was built with — ICP found the
  motion that maps the pre-op surface back onto the intra-op one.
- **`max transform diff = 0.000e+00`.** The GPU and CPU transforms are *bit-for-bit
  identical*, because the covariance reduction is done in integer fixed-point
  (see `src/icp.h`). Determinism you can see.

## Expected result

```
4.17 -- Real-Time Intraoperative / Image-Guided Surgery
ICP rigid registration: 36 moving pts -> 36 fixed pts, 12 iterations
RMS alignment error per iteration (mm):
  iter  1:   3.223807
  iter  2:   0.241553
  ...
  iter 12:   0.241553
recovered transform [R | t] (maps pre-op onto intra-op):
  [  0.97079  0.20805  0.11953 |  -5.33837 ]
  [ -0.21910  0.97171  0.08813 |   4.95205 ]
  [ -0.09781 -0.11175  0.98891 |  -2.83900 ]
final RMS error = 0.241553 mm
RESULT: PASS (GPU transform matches CPU reference)
```

(Timing lines on stderr will differ on your machine; only the stdout above is
checked.)
