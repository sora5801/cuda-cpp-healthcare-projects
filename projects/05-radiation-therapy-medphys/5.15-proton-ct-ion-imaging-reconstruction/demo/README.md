# Demo — 5.15 Proton CT & Ion Imaging Reconstruction

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/protons_sample.txt` input.
3. **Reconstruct** a relative-stopping-power (RSP) map from 1440 synthetic
   proton histories using SART on both the CPU reference and the GPU.
4. **Verify** the GPU reconstruction against the CPU reference and print a clear
   `PASS`/`FAIL`.
5. **Time** the kernels (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## Expected result

```
5.15 -- Proton CT & Ion Imaging Reconstruction
list-mode SART: 1440 protons, 40 sweeps, relax=0.80, 64 MLP samples/proton
grid: 32x32 voxels over world [-5.00,5.00]^2 cm
reconstructed RSP: center=1.1586  q1=0.9896  q3=0.7986
mean RSP inside phantom = 1.0244
RMSE vs ground-truth RSP = 0.0935
central row RSP profile (8 samples): -0.0559 0.8559 0.9255 0.8904 1.1440 1.5959 0.9487 -0.0152
RESULT: PASS (GPU matches CPU within tol=1.0e-03)
```

## How to read it

- **mean RSP inside phantom = 1.0244** — the reconstruction recovers the known
  water/insert phantom (ground-truth mean ≈ 1.057, printed on stderr) to ~3%.
- **RMSE vs ground-truth RSP = 0.0935** — whole-image root-mean-square error
  against the phantom the data was made from. It shrinks with more SART sweeps
  (an exercise in the README).
- **central row profile** — a slice through the image: ~0 outside the object,
  ~0.9–1.1 through the water, and a peak of **1.5959** where the row crosses the
  dense "bone" insert (true RSP 1.6). The small negative values just outside the
  object are the classic algebraic-reconstruction edge ringing.
- **RESULT: PASS** — the GPU RSP image matches the CPU reference within
  `1.0e-03` RSP units (the actual max error, on stderr, is ~`1e-6`).

The GPU path assigns **one thread per proton** per SART sweep and accumulates
corrections in **fixed-point integers** so the many-writer reduction is
deterministic and bit-consistent with the CPU (see `../THEORY.md`).
