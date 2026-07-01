# Demo — 5.2 Radiotherapy Treatment-Plan Optimization

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

```
5.2 -- Radiotherapy Treatment-Plan Optimization
Fluence-map optimization: 48 voxels (PTV 7, OAR 6, BODY 35), 16 beamlets, nnz=178
optimizer: projected gradient, 400 iters, step=0.020, Rx=60.0 Gy
final objective F(x) = 253.8840
PTV dose (Gy): mean 59.244  min 55.221  max 62.787  homogeneity 0.1277
OAR dose (Gy): mean 10.955  max 29.164  (tolerance-limited sparing)
RESULT: PASS (GPU plan matches CPU within dose tol=1.0e-02 Gy)
```

What the learner is seeing: the optimizer tuned 16 beamlet intensities so the
7-voxel tumor (**PTV**) reaches a mean of **59.2 Gy** against a 60 Gy
prescription, while the 6-voxel organ (**OAR**) is held to ~11 Gy mean. The
`RESULT: PASS` line confirms the GPU plan (cuSPARSE SpMVs) matches the serial CPU
reference within the documented `1e-2 Gy` dose tolerance. The **stderr** block
adds the CPU/GPU timings and the exact measured difference — those vary run to run
(and the GPU is *slower* here because the 48×16 sample is dominated by
kernel-launch and cuSPARSE-setup overhead; its edge appears only on large
matrices), so they are shown but not diffed against `expected_output.txt`.
