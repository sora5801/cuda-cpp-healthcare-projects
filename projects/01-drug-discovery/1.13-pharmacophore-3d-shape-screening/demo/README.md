# Demo — 1.13 Pharmacophore & 3D Shape Screening

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/conformers_sample.txt` input — one
   query molecule screened against 9 synthetic library conformers.
3. **Verify** the GPU Shape-Tanimoto scores against the CPU reference
   (`reference_cpu.cpp`) and print a clear `PASS`/`FAIL`.
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## How to read the result

The library conformers were **engineered** (see `scripts/make_synthetic.py`) so
the ranking recovers a known answer:

- `lib_00_self` is an **exact copy** of the query, so it must score **exactly
  `1.000000`** — a built-in correctness check you can eyeball.
- the next hits are graded perturbations: a sub-angstrom *jitter*, a 0.5 A
  *shift*, a 30° *rotation* (the symmetric ring is nearly rotation-invariant,
  so it stays high), then a *grown* variant — all in the physically sensible
  order (smaller geometric change ⇒ higher Shape Tanimoto).
- conformers placed 8 A away (`lib_07_far`) or with a totally different linear
  shape (`lib_08_line`) score near zero and fall out of the top-5.

## Expected result

```
1.13 -- Pharmacophore & 3D Shape Screening
Gaussian shape screen: query (7 atoms) vs 9 library conformers
top-5 by Shape Tanimoto:
  #1  lib_00_self  ShapeTanimoto = 1.000000
  #2  lib_01_jitter  ShapeTanimoto = 0.983953
  #3  lib_02_shift05  ShapeTanimoto = 0.942894
  #4  lib_04_rot  ShapeTanimoto = 0.930734
  #5  lib_05_grow  ShapeTanimoto = 0.864093
RESULT: PASS (GPU matches CPU within tol=1.0e-09)
```

The `[verify]` line on stderr shows `max_abs_err ≈ 3e-16` — the GPU and CPU agree
to machine precision because both run the **identical** double-precision physics
from `src/shape_overlap.h` (the shared `__host__ __device__` core).
