# Demo — 4.5 PET Image Reconstruction

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/sinogram_sample.txt` — a synthetic PET
   sinogram forward-projected from a known two-disc emission phantom with Poisson
   noise baked in.
3. **Reconstruct** the image with **MLEM** on both the CPU (reference) and the GPU,
   then **verify** the GPU image matches the CPU one within tolerance (`1e-3`) and
   print a clear `PASS`/`FAIL`.
4. **Time** the CPU and GPU MLEM loops (CUDA events) — a *teaching artifact*, not a
   benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## What to look for

Because the sinogram came from a *known* phantom, the reconstruction is
interpretable:

- **`center pixel activity`** is bright — the big central disc reconstructs to a
  high value.
- **`peak activity`** lands near `(px,py)=(21,20)` — the smaller, brighter
  off-center hot spot the phantom contains.
- **`central row profile`** is high across the middle (the disc) and near-zero at
  the edges (background) — the classic bright-object-on-dark-field shape.
- **`RESULT: PASS`** means the LOR-parallel GPU projections agree with the serial
  CPU reference.

## Expected result

```
4.5 -- PET Image Reconstruction (MLEM)
MLEM: 30 iterations, 30 angles x 45 detectors -> 32x32 image
center pixel activity = 10.7865
peak activity = 24.6882 at (px,py)=(21,20)
total reconstructed activity = 3328.4667
central row profile (8 samples): 0.0902 0.3062 11.6823 11.1942 13.2607 12.9219 0.2768 0.0290
RESULT: PASS (GPU matches CPU within tol=1.0e-03)
```

Timing (on stderr) varies per machine; on the reference RTX 2080 (`sm_75`) the tiny
32×32 problem is launch-bound, so the CPU can beat the GPU here — that is expected
and honest (see [`../THEORY.md`](../THEORY.md) "How we verify" and the timing note).
The GPU's edge grows with image size, angle count, and iteration count.

> **Note on Debug builds.** `expected_output.txt` is captured from the **Release**
> build (what `run_demo` builds). A Debug build (`-G` device debug) can differ in
> the last printed digit of one or two profile values due to different
> floating-point contraction — this is the same FMA effect documented in
> `docs/PATTERNS.md §4`. Both builds still PASS the internal GPU-vs-CPU check.
