# Demo — 4.30 Deconvolution Microscopy

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/blurred_image.txt` (a synthetic blurry
   microscopy image).
3. **Deconvolve** it with Richardson-Lucy twice — once on the **CPU** (direct
   circular convolution) and once on the **GPU** (cuFFT FFT convolution) — and
   **verify** the two restored images agree within `atol = 1e-6`. Prints `PASS`/`FAIL`.
4. **Time** the GPU iteration loop (CUDA events) and the CPU baseline — a
   *teaching artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the worst pixel error (which vary run to run),
  so it is shown but never diffed.

## Expected result

```
4.30 -- Deconvolution Microscopy (Richardson-Lucy, cuFFT)
image: 48x48 pixels   PSF: Gaussian r=4 sigma=1.5   RL iters: 30
sharpness (mean sq gradient):  blurry=2.4640  deconvolved=15.5577  (x6.31 sharper)
restored pixels along diagonal (x=y): (8,8)=2.015 (16,16)=21.873 (24,24)=28.055 (32,32)=21.889 (40,40)=2.017
total intensity:  observed=8708.00  deconvolved=8708.00
RESULT: PASS (GPU cuFFT deconvolution matches CPU reference within atol=1e-6)
```

## What to look for

- **`x6.31 sharper`** — the deconvolved image has ~6× the mean-squared-gradient of
  the blurry input. Deconvolution restored high-frequency detail (the point
  sources and edges the blur smeared out).
- **The diagonal pixels** — `(16,16)` and `(24,24)` and `(32,32)` light up: those
  are the bright synthetic beads, recovered from a blurry blob back toward sharp
  points. The corner-ish samples `(8,8)`/`(40,40)` stay near background (`~2.0`),
  as they should.
- **`total intensity ... observed=8708.00 deconvolved=8708.00`** — Richardson-Lucy
  conserves total intensity (the PSF sums to 1). The numbers match to the penny.
- **`RESULT: PASS`** — the cuFFT GPU result matched the CPU reference. On the
  stderr line you will see the worst per-pixel error is ~`1e-13` — far below the
  `1e-6` tolerance (see THEORY.md "How we verify correctness").

> The timing line is illustrative only. At this tiny 48×48 size the GPU already
> wins (FFT convolution is `O(N log N)` vs the CPU's `O(N·K)` direct convolution),
> and its edge grows with image size and PSF width.
