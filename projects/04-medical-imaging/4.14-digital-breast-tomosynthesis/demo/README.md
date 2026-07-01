# Demo — 4.14 Digital Breast Tomosynthesis

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/dbt_sample.txt` input — a synthetic
   compressed-breast slice (soft-tissue ellipse + two dense "lesion" discs)
   forward-projected over a narrow +/-25 deg angular wedge.
3. **Reconstruct** the slice with **SART** (Simultaneous Algebraic Reconstruction
   Technique) on both the CPU (reference) and the GPU, and **verify** they agree
   within tolerance (`tol = 1.0e-03`) — printing a clear `PASS`/`FAIL`.
4. **Time** the CPU reference and the GPU kernels (CUDA events) — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## Reading the result

```
4.14 -- Digital Breast Tomosynthesis
Limited-angle SART: 15 projections over +/-25.0 deg, 96 detectors -> 64x64 image
SART: 20 iterations, relaxation lambda = 0.30
center pixel value = 0.0717
peak value = 0.3479 at (px,py)=(23,31)
central row profile (8 samples): 0.0000 0.0000 0.0187 0.0000 0.0685 0.0642 0.0000 0.0000
RESULT: PASS (GPU matches CPU within tol=1.0e-03)
```

- **peak value ... at (px,py)=(23,31)** — the reconstruction's brightest pixel
  lands on **planted lesion 1** (world x = -0.28·W, y = 0). Pixel column 23 maps
  to world x = -1 + 23·(2/63) ≈ -0.27, and row 31 is essentially the central row
  (N/2 = 32). The iterative method **recovered a known dense structure** from only
  a narrow-angle wedge — the whole point of DBT.
- **central row profile** — 8 evenly spaced samples across the middle of the
  image; the non-zero humps mark where the two lesions and fibroglandular tissue
  sit. Values are low and the contrast is muted: that is the honest signature of
  **limited-angle reconstruction**, which is ill-posed and under-recovers
  attenuation compared to a full 180 deg CT scan (project 4.01).
- **GPU==CPU** — the GPU SART matches the serial CPU SART to ~2e-7, far inside the
  1e-3 tolerance, because both call the *same* per-ray math (see `src/dbt_geometry.h`).

> **Synthetic, not clinical.** The phantom and its lesions are software constructs
> in arbitrary attenuation units. Nothing here is a diagnostic image.
