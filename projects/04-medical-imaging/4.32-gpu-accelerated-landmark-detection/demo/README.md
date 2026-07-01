# Demo — 4.32 GPU-Accelerated Landmark Detection

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/heatmaps_sample.txt` (5 synthetic
   landmark heatmaps over a 20×20×16 voxel grid).
3. **Decode** each heatmap into a landmark coordinate on both the CPU reference
   and the GPU, and **verify** they agree: the integer argmax peaks match
   *exactly* and the sub-voxel soft-argmax coordinates match within `1e-9`.
4. **Report** the recovered coordinate and its error against the *planted*
   ground-truth point for each landmark (a science check), then `PASS`/`FAIL`.
5. **Time** the GPU kernel (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic (the decode is all integer /
  fixed-point math) and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing (which varies run to run), so it is shown but
  never diffed.

## Expected result

```
4.32 -- GPU-Accelerated Landmark Detection
heatmap decode: 5 landmarks over a 20x20x16 voxel grid (argmax + soft-argmax, radius 2)
  L00: peak(  6,  6,  4) val=0.9439  ->  coord=(  5.7477,  6.1897,  4.0635)  err=0.1879
  L01: peak( 12, 10,  8) val=0.9376  ->  coord=( 12.2523,  9.8732,  7.8103)  err=0.1984
  L02: peak(  8, 13, 10) val=0.9190  ->  coord=(  8.1268, 13.3142, 10.1897)  err=0.2282
  L03: peak( 15,  4, 12) val=0.9629  ->  coord=( 14.9365,  4.2523, 12.0000)  err=0.1522
  L04: peak( 10, 10,  5) val=0.9418  ->  coord=( 10.0635, 10.0635,  5.3142)  err=0.1929
worst recovery error = 0.2282 voxels
RESULT: PASS (GPU landmarks match CPU: peaks exact, coords within tol=1.0e-09)
```

## How to read it

- **`peak(x,y,z) val`** — the integer argmax voxel and its heatmap value. This is
  the *coarse* localization: the single strongest voxel in the whole volume.
- **`coord=(x,y,z)`** — the *refined*, sub-voxel coordinate from the soft-argmax
  (intensity-weighted centroid over a 5×5×5 window). Notice the fractional parts:
  they recover position the integer grid cannot represent.
- **`err`** — Euclidean distance (in voxels) between the recovered coordinate and
  the ground-truth centre the synthetic blob was built around. It is small but
  non-zero because a finite window and Gaussian tails bias the centroid slightly
  — an honest, real limitation of soft-argmax discussed in `../THEORY.md`.

> All data here is **synthetic** and for **teaching only** — no clinical use.
