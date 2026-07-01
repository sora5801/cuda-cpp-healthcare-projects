# Demo — 5.7 Brachytherapy Dose & Source Modeling

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/plan_sample.txt` (a synthetic TG-43
   plan: one Ir-192-like line source, 3 dwells, a 41×41×1 dose grid).
3. **Verify** the GPU dose against the CPU reference (`reference_cpu.cpp`) and
   print a clear `PASS`/`FAIL`. Because both sides evaluate the *identical*
   `tg43_physics.h` math with identical `double` accumulation, they agree
   **exactly** (`max_rel_err = 0`).
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run) plus
  the source path, so it is shown but never diffed.

## Reading the result

- **`max dose … at voxel (20,20,0)`** — the hottest voxel is the grid center,
  exactly where the dwells sit: dose diverges ~`1/r²` as `r→0`, so the on-source
  voxel is by far the largest. This is the expected (and honestly non-physical)
  behavior right at the source; see THEORY §"Numerical considerations".
- **`center-row profile`** — 8 evenly spaced samples across the central row. The
  values rise toward the middle and fall symmetrically outward — the visible
  `1/r²` falloff of a brachytherapy source.
- **`dose @ ~1cm transverse probe`** — dose 1 cm off the source cluster, a
  clinically meaningful reference distance for TG-43.

## Expected result

```
5.7 -- Brachytherapy Dose & Source Modeling
[TG-43 analytic dose | SYNTHETIC teaching source, not clinical]
source: line L=0.35 cm  Lambda=1.1090 cGy/(h*U)  dwells=3
grid: 41 x 41 x 1 voxels @ 0.10 cm  (1681 voxels)
max dose = 98511.132812 cGy/h at voxel (20,20,0)
dose @ ~1cm transverse probe = 2.818497 cGy/h
center-row profile (8 samples): 0.766199 1.339740 3.383707 16.877193 29.011187 4.121541 1.526351 0.766199
RESULT: PASS (GPU matches CPU within rel-tol=1.0e-05)
```

(Your machine's `stderr` timing line will differ; that is expected and not
diffed.)
