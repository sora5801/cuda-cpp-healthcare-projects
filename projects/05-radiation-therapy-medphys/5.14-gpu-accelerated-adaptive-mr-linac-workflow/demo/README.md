# Demo — 5.14 GPU-Accelerated Adaptive MR-Linac Workflow

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** the reduced-scope online-adaptive-radiotherapy (oART) workflow on the
   committed `data/sample/oart_case.txt`:
   - **register** the synthetic *daily* MR to the *planning* MR (Demons + Gaussian
     smoothing),
   - **warp the planned dose** onto the daily anatomy,
   - **score** plan-approval metrics over the GTV (mean, D95, coverage).
3. **Verify** the GPU result against the CPU reference (`reference_cpu.cpp`): the
   displacement field, the warped dose, and the scalar metrics must all agree
   within tolerance. Prints a clear `PASS`/`FAIL`.
4. **Time** the GPU kernels (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timings and the GPU-vs-CPU numeric error (which vary run
  to run), so it is shown but never diffed.

## What the numbers mean

- `MSE before / after` — mean-squared intensity mismatch between the moving and
  fixed images, before vs after registration. The **big drop** (here 0.0245 →
  0.000069, ~356×) is the registration working: the daily anatomy has been aligned
  back onto the planning anatomy.
- `peak displacement magnitude` — the largest voxel shift in the recovered field.
  The synthetic ground-truth motion is `(3, 2)` voxels (magnitude ~3.6); the field
  overshoots at the low-signal image borders (a real, expected Demons artifact),
  but the tumour region — where the MSE is measured — converges tightly.
- `GTV plan metrics on WARPED dose` — the dose actually delivered to the daily
  tumour after adaptation: `mean`, `D95` (dose covering ≥95% of GTV voxels), and
  `coverage` (fraction of GTV at or above `dose_thresh`). This is the "did the plan
  still hit the target?" check that a physician approves in the clinic.

## Expected result

```
5.14 -- GPU-Accelerated Adaptive MR-Linac Workflow
[reduced-scope teaching version: 2-D Demons registration + dose warp; synthetic data]
case: 32x32 voxels, 60 Demons iters, sigma=1.50, K=1.00, dose_thresh=30.00 Gy
registration MSE(moving vs fixed): before=0.024537  after=0.000069
peak displacement magnitude: 7.110311 voxels
GTV plan metrics on WARPED dose:  mean=42.056072 Gy  D95=21.902282 Gy  coverage(>=30.00 Gy)=0.786517
RESULT: PASS (GPU matches CPU within tol=1.0e-06)
```

> All data here is **synthetic** and carries **no clinical meaning**. This is a
> reduced-scope teaching version of a research-grade clinical pipeline; see
> [`../THEORY.md`](../THEORY.md) for what is simplified and how the real oART chain
> differs.
