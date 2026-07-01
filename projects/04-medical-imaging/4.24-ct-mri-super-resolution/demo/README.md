# Demo — 4.24 CT/MRI Super-Resolution

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/phantom_hr.txt` (a synthetic
   ground-truth high-res image).
3. The program **degrades** that image 2× to a low-res input, **super-resolves**
   it back on both the CPU and the GPU, and **verifies** the GPU result against
   the CPU reference (`reference_cpu.cpp`) with a clear `PASS`/`FAIL`.
4. It reports **PSNR** of the super-resolved image vs. the ground truth, and vs.
   a naive nearest-neighbour upscaling — so you can see the learned network add
   real image quality (higher dB).
5. It **times** the kernel (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt): the PSNR numbers, 8 evenly-spaced
  HR pixel values, and the `RESULT:` line.
- **stderr** carries the timing and the max-abs-error (which vary run to run), so
  it is shown but never diffed.

## Expected result (captured from a real run on an RTX 2080, sm_75)

```
4.24 -- CT/MRI Super-Resolution
scale R=2  |  LR 16x16 -> HR 32x32  |  net: 4 feat ch, 3x3 conv + subpixel
PSNR nearest-neighbour vs truth = 22.5583 dB
PSNR super-resolved   vs truth = 23.7905 dB
PSNR improvement over baseline = 1.2322 dB
HR samples (8 evenly spaced): 0.198325 0.821572 0.512523 0.610395 0.660512 0.370428 0.288678 0.155375
RESULT: PASS (GPU matches CPU within tol=1.0e-06)
```

The **positive PSNR improvement** is the headline: the sub-pixel-conv network
reconstructs a sharper image than block-replication. The `RESULT: PASS` line
confirms the GPU kernel and the CPU reference agree (they call the *same*
`sr_hr_pixel()` in `sr_core.h`, so they match to ~1e-6). Your PSNR numbers should
be identical (deterministic); only the stderr timing changes between runs.
