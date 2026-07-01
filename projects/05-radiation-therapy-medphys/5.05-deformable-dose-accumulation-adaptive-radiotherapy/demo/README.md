# Demo — 5.5 Deformable Dose Accumulation & Adaptive Radiotherapy

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** the full adaptive-radiotherapy mini-pipeline on the committed
   `data/sample/art_case.txt`:
   - **DIR** — register today's anatomy (`daily_img`) onto the planning anatomy
     (`plan_img`) with GPU Demons → a displacement vector field (DVF).
   - **Deformable dose warp + accumulation** — map the delivered daily dose back
     through the DVF into the planning frame and sum it over 3 fractions
     (summation of deformed doses).
   - **DVH** — build the dose-volume histogram of the accumulated dose.
3. **Verify** the GPU result against the CPU reference (`reference_cpu.cpp`) on
   three independent checks and print a clear `PASS`/`FAIL`:
   - DVF agrees to ≤ 1e-3 px,
   - accumulated dose agrees to ≤ 1e-9 Gy,
   - the (integer) DVH matches **exactly**.
4. **Time** the CPU pipeline and the two GPU stages (CUDA events) — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the exact GPU-vs-CPU error (which vary run to
  run), so it is shown but never diffed.

## Expected result

```
5.5 -- Deformable Dose Accumulation & Adaptive Radiotherapy
ART pipeline: 64x64 grid | DIR 120 iters, sigma=1.50 px | 3 fractions
DIR: mean |displacement| = 4.5079 px
accumulated dose: sum = 4232.1868 Gy*vox, hot-spot = 5.9880 Gy
rigid (no-DIR) hot-spot = 6.0000 Gy  (deformable - rigid = -0.0120 Gy)
cumulative DVH (dose_Gy: vol%):  0.00:100.0  0.75:35.6  1.50:24.0  2.25:17.1  3.00:12.3  3.75:8.3  4.50:5.1  5.25:2.4
RESULT: PASS (DVF<=1e-3px, dose<=1e-9Gy, DVH exact)
```

## How to read it

- **mean |displacement| = 4.51 px** — the DVF the Demons solver recovered; it
  matches the shift+stretch baked into the synthetic daily image (~4.5 px), so the
  registration is correct.
- **accumulated dose hot-spot = 5.99 Gy** vs **rigid hot-spot = 6.00 Gy** — three
  fractions at ~2 Gy each. The small gap (`deformable − rigid = −0.012 Gy`) is the
  point of the whole project: warping each fraction's dose through the DVF before
  summing gives a *different* total than naively adding it in place. Here the dose
  cloud is broad and smooth, so the discrepancy is small — with a steeper dose
  gradient or larger motion it grows (try `--shift 12.0`), which is exactly why
  clinics use deformable, not rigid, accumulation.
- **cumulative DVH** — for each listed dose (Gy), the percent of voxels receiving
  **at least** that dose. `0.00:100.0` (every voxel gets ≥ 0 Gy) down to
  `5.25:2.4` (only the hot core exceeds 5.25 Gy) traces the falloff of the cloud.
- **RESULT: PASS** — the GPU reproduced the serial CPU pipeline within tolerance
  on all three checks (DVF, dose, DVH).
