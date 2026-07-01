# Demo — 4.16 Functional MRI Analysis

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/fmri_sample.txt` (48 synthetic voxels ×
   80 scans of a simulated block-design task-fMRI experiment).
3. **Fit the GLM** at every voxel on both the CPU and the GPU, and **verify** the
   GPU t-statistics against the CPU reference (`reference_cpu.cpp`), printing a
   clear `PASS`/`FAIL`.
4. **Report** the top-6 voxels by task t-statistic and how many of them are truly
   task-active (recovering the planted ground truth).
5. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt). The t-statistics are printed to a
  fixed 4 decimals so the comparison is stable.
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## Expected result

```
4.16 -- Functional MRI Analysis
mass-univariate GLM: 48 voxels x 80 scans (TR=2.0s, block=10 scans)
design: [task(HRF), linear-drift, intercept]   contrast = task
top-6 voxels by task t-statistic (tie -> lower index):
  #1  voxel[  0]  t =  17.7342   beta_task =   6.3126   [active]
  #2  voxel[ 36]  t =  17.4347   beta_task =   6.0117   [active]
  #3  voxel[ 18]  t =  16.1324   beta_task =   5.6646   [active]
  #4  voxel[ 24]  t =  15.4637   beta_task =   5.6590   [active]
  #5  voxel[ 12]  t =  15.2087   beta_task =   6.2369   [active]
  #6  voxel[ 30]  t =  15.0395   beta_task =   5.7055   [active]
recovered 6/6 top voxels that are truly task-active (of 8 active total)
RESULT: PASS (GPU matches CPU within tol=1.0e-09)
```

## How to read it

- Each `voxel[v]` line is one independent ordinary-least-squares GLM fit. `t` is
  the **task t-statistic** (bigger = stronger, more reliable activation);
  `beta_task` is the fitted amplitude of the HRF-convolved task regressor.
- The `[active]` tag is the synthetic ground truth (see `data/README.md`). The GLM
  never reads it — the fact that all 6 top-t voxels carry that tag is what
  "recovered 6/6" is telling you: the pipeline found exactly the voxels we planted.
- On this tiny sample the GPU and CPU times are similar because the run is
  dominated by launch/copy overhead — the honest-timing lesson (docs/PATTERNS.md
  §7). The GPU's advantage appears at whole-brain scale (V ~ 10⁵ voxels).
