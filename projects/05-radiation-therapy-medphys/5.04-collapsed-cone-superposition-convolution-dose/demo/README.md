# Demo — 5.4 Collapsed-Cone / Superposition-Convolution Dose

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed synthetic phantom (`data/sample/phantom.txt`).
3. **Verify** the GPU dose grid against the CPU reference (`reference_cpu.cpp`):
   the integer dose grids must match **exactly** (0 mismatches), and the
   double-precision TERMA within a tiny tolerance. Prints a clear `PASS`/`FAIL`.
4. **Time** both stages (CUDA events for the GPU, `std::chrono` for the CPU) — a
   *teaching artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt). Determinism comes from
  accumulating dose as **integer dose-units** so the GPU's `atomicAdd` scatter is
  order-independent (PATTERNS.md §3).
- **stderr** carries timing and the numeric error (which vary run to run), so it
  is shown but never diffed.

## Reading the result: a percent-depth-dose (PDD) curve

The key line is the **central-axis depth-dose**: integer dose-units down the
middle beam column, one number per row (surface → depth). This is the classic PDD
curve of radiotherapy physics. In the synthetic phantom (water / lung / water /
bone / water) you can see:

- **Build-up** near the entry surface (dose rises from the surface).
- A **dip through the low-density lung** (rows 4–7): less mass, less local dose,
  but the scatter kernel reaches *farther*.
- A **pile-up at the bone interface** (rows 12–13): high density deposits more.

That density-dependent reshaping is exactly what a naive "dose falls off with
depth-in-water" model gets wrong and what superposition-convolution gets right.

## Expected result

```
5.4 -- Collapsed-Cone / Superposition-Convolution Dose
grid 16x16 voxels @ 0.50 cm, mu/rho=0.0600 cm^2/g, 8 cones, a=1.200 /(g/cm^2)
beam columns [6..9], dose_scale=1000000 units/dose
total deposited = 281592180 dose-units; peak voxel (x=7,y=3)
central-axis depth-dose (column x=7), dose-units per row y=0..15:
  2640504 3833134 4253406 4689390 1480281 1534496 1524208 1449695 4495404 3949322 3643425 3296351 4415469 3981154 2422857 1712640
RESULT: PASS (GPU dose grid matches CPU exactly; TERMA within 1e-09)
```

Timing (on stderr) varies per machine; on this tiny grid the GPU is launch-bound
and slower than the CPU — the GPU's edge appears at clinical scale (512³ voxels ×
~400 cones), as noted in the stderr timing line and `THEORY.md`.
