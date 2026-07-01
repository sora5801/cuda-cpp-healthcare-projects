# Demo — 4.23 Arterial Spin Labeling & Perfusion Imaging

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/asl_sample.txt` (6 voxels × 7 post-labeling delays).
3. **Fit** the Buxton kinetic model per voxel — on the CPU (reference) and on the
   GPU (one thread per voxel) — recovering cerebral blood flow (CBF) and arterial
   transit time (ATT).
4. **Verify** two things: the GPU per-voxel fits match the CPU to ~machine
   precision, AND the fit recovers the known ground-truth physiology used to
   synthesize the noise-free curves.
5. **Time** the fit (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately (docs/PATTERNS.md §3):

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric errors (which vary run to run), so
  it is shown but never diffed.

## Expected result

```
4.23 -- Arterial Spin Labeling & Perfusion Imaging
multi-delay ASL Buxton fit: 6 voxels x 7 PLDs, Levenberg-Marquardt (<=30 it)
PLDs (s): 0.50 0.75 1.00 1.25 1.50 2.00 2.50
per-voxel fit (true -> recovered):
  vox   CBF_true CBF_fit   ATT_true ATT_fit   iters
  v0      60.000  60.000      0.700   0.700       3
  v1      55.000  55.000      1.000   1.000       5
  v2      22.000  22.000      1.200   1.200       5
  v3      18.000  18.000      1.400   1.400       6
  v4      80.000  80.000      0.500   0.500       6
  v5      40.000  40.000      0.900   0.900       5
mean recovered: CBF = 45.833 mL/100g/min   ATT = 0.950 s
RESULT: PASS (GPU==CPU within 1e-09; fit recovers ground truth within 1e-04)
```

Each voxel's `CBF_fit`/`ATT_fit` reproduce its `CBF_true`/`ATT_true` to the printed
precision because the sample curves are **noise-free** — the Levenberg-Marquardt
optimizer drives the residual to ~1e-8. `RESULT: PASS` means both the GPU==CPU
check (~7e-15) and the ground-truth recovery check (~1e-8) passed. On stderr you
will see the GPU is *slower* here than the CPU: with only 6 voxels the kernel is
launch-bound (one launch, negligible work) — the GPU's advantage appears at
whole-brain scale (10⁵–10⁶ voxels), where the CPU's serial loop dominates.

> The data are **synthetic** and the model is a single-compartment teaching
> reduction — a software demonstration of the kinetic fit, not a perfusion
> measurement, and not for any clinical use.
