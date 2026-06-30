# Demo — 4.28 GPU-Accelerated DRR Generation for 2D/3D Registration

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Render a DRR** — a simulated X-ray — of the committed synthetic CT phantom in
   `data/sample/`, on the CPU (reference) and on the GPU (one thread per detector
   pixel, each marching a ray through the volume).
3. **Verify** the GPU image against the CPU reference and print a clear
   `PASS`/`FAIL` (they agree to ~1e-6 because both call the *same* `integrate_ray`).
4. **Time** the GPU kernel (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## Expected result

```
4.28 -- GPU-Accelerated DRR Generation for 2D/3D Registration
CT volume: 32x32x32 voxels, spacing 2.00x2.00x2.00 mm
DRR detector: 128x128 pixels, ray step 1.00 mm (cone-beam, lateral view)
center pixel attenuation = 1.2518
max attenuation = 1.3561 at (u,v)=(76,61)
central row profile (8 samples): 0.0000 0.2550 0.8656 1.0276 1.3355 0.8137 0.0202 0.0000
RESULT: PASS (GPU matches CPU within tol=1.0e-03)
```

## How to read it

- **center pixel attenuation = 1.2518** — the line integral ∑μ·ds (dimensionless)
  for the ray through the detector center. The Beer–Lambert transmitted intensity
  would be `I = I0·exp(−1.2518)`; we report the integral itself because it is the
  quantity intensity-based registration compares.
- **max attenuation … at (u,v)=(76,61)** — the brightest DRR pixel. The phantom's
  dense bone sphere is **offset** toward +x/+y, so the brightest pixel is *right of
  center* (column 76 > the 64-px midline). That off-center peak is the visual proof
  the cone-beam ray geometry is correct.
- **central row profile** — 8 samples across the middle detector row: 0 at the air
  edges, rising through the soft-tissue body, peaking (1.3355) where the ray clips
  the bone sphere. A 1-D slice of the radiograph.
- The **stderr** line shows the CPU render (~170–240 ms) vs the GPU kernel
  (~0.6 ms) — a large speedup that grows with panel size and the 50–200 DRRs a real
  registration needs per iteration.

Everything here is **synthetic** and **educational** — not a real radiograph and
not for any clinical use.
