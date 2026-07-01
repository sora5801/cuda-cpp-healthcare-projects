# Demo — 4.9 Image Denoising & Restoration (Non-Local Means)

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/phantom_sample.txt` — a 32×32
   synthetic phantom with additive Gaussian noise (plus its clean ground truth).
3. **Denoise** it with Non-Local Means on both the CPU reference
   (`src/reference_cpu.cpp`) and the GPU kernel (`src/kernels.cu`), then **verify**
   the two agree within `1.0e-4` and print a clear `PASS`/`FAIL`.
4. **Report** PSNR before/after denoising (so you can see the noise actually
   removed) and **time** the kernel (CUDA events) vs the CPU baseline — a
   *teaching artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt). The PSNR values and pixel samples
  are computed from the CPU reference (a fixed-order computation), so they are
  stable across machines.
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## Expected result

```
4.9 -- Image Denoising & Restoration (Non-Local Means)
Non-Local Means denoise: 32x32 image, patch r=2, search r=5, sigma=0.080, h=0.096
PSNR noisy  vs clean = 22.0857 dB
PSNR denoised vs clean = 29.9900 dB
PSNR improvement = 7.9043 dB
denoised central row (8 samples): 0.1317 0.1689 0.7330 0.7455 0.7630 0.7643 0.6591 0.1688
RESULT: PASS (GPU matches CPU within tol=1.0e-04)
```

## How to read it

- **PSNR improvement +7.90 dB** — the denoiser lifted image quality from 22.1 dB to
  30.0 dB against the clean ground truth. Higher PSNR = closer to clean.
- **The central-row profile** samples 8 evenly spaced columns of the middle row of
  the denoised image. You can see the dark background (~0.13–0.17), the bright
  central disk (~0.73–0.76), and the darker square inset (the `0.6591` dip) — the
  phantom's structure recovered, with edges preserved rather than blurred.
- **RESULT: PASS** — the GPU image matched the CPU reference to `~2.4e-7` (shown on
  stderr), far inside the `1.0e-4` tolerance. The tiny residual is float FMA
  contraction, explained in [../THEORY.md](../THEORY.md) §5.

The stderr timing line (e.g. `CPU denoise: 17 ms   GPU denoise: 0.4 ms`) is a
teaching artifact: on this tiny 32×32 sample the GPU is already faster, and its
edge grows fast with image size because NLM cost scales as `O(P·S²·R²)`.
