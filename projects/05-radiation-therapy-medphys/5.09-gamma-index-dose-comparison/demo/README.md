# Demo — 5.9 Gamma-Index Dose Comparison

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/dose_pair.txt` input (a synthetic
   32×32 reference/evaluated dose pair).
3. **Verify** the GPU gamma map against the CPU reference (`reference_cpu.cpp`)
   and print a clear `PASS`/`FAIL`.
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately (PATTERNS.md §3):

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt). The pass-rate is printed from
  integer counts and the gamma-map slice as integer milli-gamma (γ×1000), so no
  float-formatting drift can break the diff.
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## How to read the result

- **`analyzed points : 888`** — voxels above the 10% low-dose threshold that were
  scored (background is excluded).
- **`gamma pass-rate : 99.7 %`** — 885 of 888 analyzed voxels have γ ≤ 1. The 3
  failing voxels are the deliberately injected central hot spot (+12%, outside
  3%/3 mm).
- **`gamma min / max : 0.051 / 1.141`** — γ_max > 1 confirms the hot spot fails;
  the passing background sits well below 1.
- **`gamma[row 16, cols 12..19] (x1000)`** — a slice of the center row of the
  gamma map (γ×1000). Values ≈ 455–500 are the passing, slightly-biased
  background. (The hot spot is off this row, near col 19 / row 14.)
- **`RESULT: PASS`** — the GPU gamma map equals the CPU map within `1e-6`
  (observed error is exactly 0) and the pass-rate statistics match exactly.

## Expected result

```
5.9 -- Gamma-Index Dose Comparison
grid: 32 x 32 voxels @ 2.0 mm   criterion: 3%/3 mm   low-dose cutoff: 10%
analyzed points : 888
passing (g<=1)  : 885
gamma pass-rate : 99.7 %
gamma min / max : 0.051 / 1.141
gamma[row 16, cols 12..19] (x1000): 455 477 492 500 500 492 477 455
RESULT: PASS (GPU gamma map matches CPU within tol=1e-06)
```

The `stderr` timing/verify lines (shown by the demo, not diffed) look like:

```
[timing] CPU reference: 0.14 ms   GPU kernel: 0.40 ms
[verify] max_abs_err(gamma map) = 0.000000e+00  (tolerance 1.0e-06)
[verify] stats match (analyzed/passed/rate): yes
```

> All data here is **synthetic** and for teaching only — no clinical validity.
