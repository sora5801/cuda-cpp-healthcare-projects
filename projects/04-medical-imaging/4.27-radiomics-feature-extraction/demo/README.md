# Demo — 4.27 Radiomics Feature Extraction

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/radiomics_sample.txt` ROI volume.
3. **Verify** the GPU features against the CPU reference (`reference_cpu.cpp`):
   the integer GLCM/histogram counts must match **exactly**, and every derived
   feature within `1e-9`. Prints a clear `PASS`/`FAIL`.
4. **Time** the kernels (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt). Determinism comes for free here
  because the GLCM is built with **integer** `atomicAdd` (commutative → order
  independent → reproducible), and every printed feature is at fixed precision.
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## Expected result

```
4.27 -- Radiomics Feature Extraction
ROI: 6 x 6 x 5 grid, 56 masked voxels, 8 gray levels
intensity range [23.6925, 103.0230]
first-order:
  mean        = 2.928571
  variance    = 4.780612
  energy      = 748.000000
  entropy     = 2.727397 bits
GLCM texture (13 directions, symmetric, 820 pairs):
  contrast    = 10.390244
  energy(ASM) = 0.023308
  homogeneity = 0.278846
  correlation = -0.056568
  entropy     = 5.669793 bits
RESULT: PASS (GPU features match CPU; GLCM counts identical)
```

## How to read it

- **56 masked voxels** — the spherical ROI inside the `6×6×5` grid; background is
  ignored.
- **820 GLCM pairs** — every in-ROI neighbour pair over the 13 directions, counted
  symmetrically. This integer is identical CPU-vs-GPU (that is the exact check).
- **contrast 10.39, homogeneity 0.28, correlation −0.057** — the fingerprint of a
  **checkerboard texture**: many large intensity jumps between neighbours (high
  contrast, low homogeneity) and a slight anti-correlation. Change the synthetic
  generator to a smooth blob (drop the ripple) and watch contrast fall and
  homogeneity/correlation rise — a hands-on way to *feel* what these features mean.
