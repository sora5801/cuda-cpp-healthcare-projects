# Demo — 4.29 Light-Sheet Microscopy Reconstruction

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed synthetic plane in `data/sample/lsfm_sample.txt`.
3. **Reconstruct** the blurry image with **Richardson-Lucy deconvolution**, doing
   every convolution in the **Fourier domain with cuFFT** on the GPU.
4. **Verify** the GPU (cuFFT) result against the CPU reference (a direct DFT of
   the same math) and print a clear `PASS`/`FAIL`.
5. **Time** the GPU vs the CPU baseline — a *teaching artifact*, not a benchmark.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## How to read the result

```
input  (blurry)   : sum=69.1963  max=0.359447  L2=2.549701
output (deblurred): sum=69.1963  max=0.786365  L2=3.173884
sharpening        : peak x2.1877  L2 x1.2448  (flux ratio 1.0000)
```

- **sum** (total intensity / flux) is **unchanged** to 4 decimals: Richardson-Lucy
  is a multiplicative, flux-conserving update, so deconvolution redistributes
  light without creating or destroying it. Seeing `flux ratio 1.0000` is a quick
  sanity check that the algorithm is behaving.
- **max** rises **2.19x** and **L2** rises **1.24x**: the blur that had smeared each
  bright "bead" across neighbouring pixels is being pulled back into sharp peaks.
  That is exactly what a deconvolution should do — recover contrast lost to the
  microscope's point-spread function.

## Expected result

```
4.29 -- Light-Sheet Microscopy Reconstruction
Richardson-Lucy deconvolution (cuFFT, Fourier domain): 32x32 image, PSF sigma=1.60 px, 12 iterations
input  (blurry)   : sum=69.1963  max=0.359447  L2=2.549701
output (deblurred): sum=69.1963  max=0.786365  L2=3.173884
sharpening        : peak x2.1877  L2 x1.2448  (flux ratio 1.0000)
RESULT: PASS (GPU cuFFT matches CPU DFT within rel tol=1.0e-09)
```

The GPU and CPU paths both run in **double precision** and share the per-pixel RL
math (`src/rl_core.h`) and the PSF, so they agree to ~1e-15 (reported on stderr);
the 1e-9 tolerance is a comfortable, honest floor. See `../THEORY.md`
"Numerical considerations" for why it is not exactly bit-identical.
