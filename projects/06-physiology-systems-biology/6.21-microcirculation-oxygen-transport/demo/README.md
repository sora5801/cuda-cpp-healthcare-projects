# Demo — 6.21 Microcirculation & Oxygen Transport

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed synthetic sample
   (`data/sample/microvessel_network.txt`).
3. **Verify** the GPU oxygen field against the CPU reference (`reference_cpu.cpp`)
   and print a clear `PASS`/`FAIL`.
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## Expected result

```
6.21 -- Microcirculation & Oxygen Transport
Tissue grid: 12 x 12 x 8 = 1152 points, spacing 5.0 um; 18 capillary sources
PO2 field (mmHg): min 8.4030  mean 18.3419  max 35.9173
hypoxic fraction (PO2 < 10.0 mmHg): 3.47% (40 / 1152 points)
sample PO2 (mmHg) at fixed grid points:
  origin     idx 0     (0.0,0.0,0.0) um: 13.7819
  x-edge     idx 11    (55.0,0.0,0.0) um: 13.2133
  mid-index  idx 576   (0.0,0.0,20.0) um: 15.1098
  centre     idx 654   (30.0,30.0,20.0) um: 23.9024
  far-corner idx 1151  (55.0,55.0,35.0) um: 8.4030
RESULT: PASS (GPU field matches CPU within tol=1.0e-09)
```

## How to read it

- The **PO2 field** row is the whole-tissue summary: the minimum, mean and maximum
  oxygen partial pressure over all 1152 grid points.
- The **hypoxic fraction** counts how much of the tissue sits below a ~10 mmHg
  hypoxia threshold. Here ~3.5% of the block is hypoxic — and it is not random: it
  clusters in the corner farthest from every capillary.
- The **sample points** make that spatial story concrete: the well-perfused
  `centre` is at ~24 mmHg, while the `far-corner` (max x, y, z, far from all three
  capillaries) is the global minimum at ~8.4 mmHg. This is the classic
  "watershed / Krogh corner" hypoxia the model exists to reveal.
- `RESULT: PASS` means the GPU field matched the CPU reference to within 1e-9 mmHg
  (they actually agree to ~1e-14; both run the identical `solve_point()` math in
  the identical summation order — see `../THEORY.md` "How we verify correctness").

The stderr timing line is a teaching artifact: on this tiny 1152x18 problem the
CPU and GPU are both sub-millisecond and dominated by launch/copy overhead. The
GPU's advantage in the O(N_grid x N_src) direct sum grows with grid and source
count; a production solver replaces that O(N^2) sum with a fast multipole method.
