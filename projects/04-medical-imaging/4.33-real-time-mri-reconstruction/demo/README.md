# Demo — 4.33 Real-Time MRI Reconstruction

## What this demonstrates

One command reconstructs a **6-frame real-time MRI movie** from a synthetic
**golden-angle radial** k-space acquisition, on both the CPU and the GPU, and proves
they agree. It shows the whole non-Cartesian reconstruction pipeline the catalog asks
for — density compensation → **Kaiser-Bessel gridding** → **cuFFT** inverse FFT →
deapodization — inside a **sliding window** that slides over the streamed spokes to
produce successive frames.

## Run it

```powershell
# Windows (PowerShell) — builds Release if needed, runs, diffs stdout vs expected
./demo/run_demo.ps1
```
```bash
# Linux/macOS — uses the optional CMake build
./demo/run_demo.sh
```

## What you should see (annotated)

```
4.33 -- Real-Time MRI Reconstruction
golden-angle radial NUFFT recon (single slice, single coil), gridding + cuFFT
grid: 32x32   spokes: 64 x 64 readout   KB width: 4          <- the acquisition
sliding window: 21 spokes/frame, stride 8, 6 frames          <- the real-time plan
recon movie: normalized to peak=1.000 (raw peak 1.837e-04, arbitrary MR units)
per-frame peak (normalized) @ (row,col):
  frame 0: 0.9796 @ (14,13)                                   <- the moving blob...
  frame 1: 0.9933 @ (14,13)
  frame 2: 0.9210 @ (14,13)
  frame 3: 0.9331 @ (13,13)
  frame 4: 0.9881 @ (12,13)
  frame 5: 1.0000 @ (12,13)                                   <- ...drifts up 2 pixels
last-frame vs truth: normalized correlation = 0.9613          <- the anatomy is recovered
RESULT: PASS (GPU gridding+cuFFT matches CPU within tol; recon recovers truth)
```

- **The peak LOCATION moves** from row 14 → 13 → 12 across the frames. That is the
  synthetic "heartbeat" (one blob slowly bobs), reconstructed by the sliding window —
  the whole point of *real-time* MRI. The brightness is normalized so the numbers are
  easy to read (the raw MR-unit scale is arbitrary after gridding).
- **`correlation = 0.9613`** is the science check: the last frame's image matches the
  known synthetic phantom, i.e. gridding genuinely recovered the anatomy from only 21
  sparse radial spokes.
- **`RESULT: PASS`** requires *both* that the GPU movie matches the CPU movie within
  tolerance *and* that the reconstruction recovers the truth.

## stdout vs stderr

`stdout` (above) is **deterministic** and is what `run_demo` diffs against
[`expected_output.txt`](expected_output.txt). Timing and the exact GPU-vs-CPU error go
to **stderr** (shown but not diffed) because they vary run to run:

```
[timing] CPU movie: 6.0 ms   GPU movie (gridding+cuFFT): 4.7 ms
[verify] GPU-vs-CPU movie RMS diff = 1.65e-11  (tolerance 1.00e-04)
```

The GPU-vs-CPU difference is ~`1e-11` (essentially exact): the gridding scatter uses
**fixed-point integer atomics**, so it is bit-identical on both sides, leaving only the
cuFFT-vs-radix-2-FFT rounding. The timing is a **teaching artifact, not a benchmark** —
on a 32×32 slice the per-frame kernels are launch-bound; the GPU's advantage grows with
grid size, spoke count, and receive coils, and a production system overlaps
reconstruction with acquisition using CUDA streams.
