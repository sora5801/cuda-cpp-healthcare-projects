# Demo — 2.4 Cryo-ET Subtomogram Averaging

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/` input.
3. **Verify** the GPU result against the CPU reference (`reference_cpu.cpp`) and
   print a clear `PASS`/`FAIL`.
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## Expected result

stdout (diffed against [`expected_output.txt`](expected_output.txt)):

```
2.4 -- Cryo-ET Subtomogram Averaging
Subtomogram averaging: 1 reference vs 6 candidates, 16^3 voxels, 12 trial angles (cuFFT cross-correlation)
per-candidate alignment (best angle index : peak NCC):
  cand[0]  angle[1] =   30.0 deg   peak NCC = 0.9651
  cand[1]  angle[3] =   90.0 deg   peak NCC = 0.9681
  cand[2]  angle[5] =  150.0 deg   peak NCC = 0.9653
  cand[3]  angle[7] =  210.0 deg   peak NCC = 0.9657
  cand[4]  angle[9] =  270.0 deg   peak NCC = 0.9678
  cand[5]  angle[11] =  330.0 deg   peak NCC = 0.9655
refined average core intensity (mean|voxel|) = 0.097395
RESULT: PASS (GPU cuFFT correlation matches CPU direct, same poses)
```

### What you are seeing

- Each candidate is matched to the reference over **12 trial in-plane rotations**;
  the GPU computes the cross-correlation over **all 4096 translational shifts** in
  Fourier space (cuFFT) and reports the **best** (`peak NCC ≈ 0.965`).
- The recovered angle indices `[1, 3, 5, 7, 9, 11]` are exactly the angles
  **planted** in the synthetic data (`data/README.md`) — the alignment search
  found the right poses.
- The **refined average core intensity** is the single deterministic scalar
  summarizing the averaged (aligned) cube; CPU and GPU agree to the last digit.

On **stderr** (shown, not diffed) you also see the CPU-vs-GPU timing and the
verification numbers: the worst zero-shift NCC error between the GPU's cuFFT path
and the CPU's direct sum (`≈ 3.2e-07`, well within the `1e-3` tolerance), and a
confirmation that both paths chose the same best angle for every candidate.
