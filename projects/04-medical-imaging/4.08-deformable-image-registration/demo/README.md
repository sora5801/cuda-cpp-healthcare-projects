# Demo — 4.8 Deformable Image Registration

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/dir_pair.txt` (a synthetic fixed/moving image pair).
3. **Register** the moving image onto the fixed one with Thirion's Demons, on **both** the GPU
   (`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`).
4. **Verify** the two displacement fields agree within `1e-3` px, and report the **SSD before vs. after** —
   the number that shows the registration actually worked.
5. **Time** the GPU loop (CUDA events) and the CPU baseline — a *teaching artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the exact GPU-vs-CPU numeric error (which vary run to run), so it is shown
  but never diffed.

## Expected result

```
4.8 -- Deformable Image Registration
Demons DIR: 64x64 image, 120 iters, sigma=1.50 px (radius=5)
SSD before = 51.8191
SSD after  = 0.0643  (99.88% reduction)
mean |displacement| = 4.5079 px
u_x along center row (8 samples): 1.3479 2.2833 3.3400 4.3975 5.3793 6.4929 8.0949 10.5250
RESULT: PASS (GPU field matches CPU within tol=1.0e-03 px)
```

## How to read it

- **SSD before / after** — the sum of squared intensity differences between the fixed image and the
  (warped) moving image. Falling from ~51.8 to ~0.06 (a **99.9% reduction**) means the moving blob snapped
  onto the fixed one. This is the *science* check.
- **mean |displacement|** — the average length of the recovered per-pixel arrows (~4.5 px), which matches the
  shift+stretch built into the synthetic sample (~5 px). The field found the right motion.
- **u_x along center row** — the horizontal displacement sampled left→right across the middle row. It rises
  smoothly and monotonically, exactly what a rightward shift plus an x-stretch should produce (larger
  correction toward the right edge). Watching this profile is the quickest way to *see* that the DVF is
  spatially varying, not a single global shift.
- **RESULT: PASS** — the GPU and CPU displacement fields agree to ~5e-15 px (reported on stderr), far under
  the 1e-3 px tolerance. Because both sides run the identical `src/demons.h` math, this is a strong
  correctness check. The exit code is 0 on PASS, which gates the demo script.

The **stderr** timing line typically shows the GPU loop finishing several times faster than the CPU baseline
on this tiny image; the gap grows with image/volume size (see THEORY §4).
