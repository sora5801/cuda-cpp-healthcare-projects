# Demo — 2.24 SAXS / SANS Data-Driven Structure Modeling

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/saxs_sample.txt` input (a synthetic
   40-atom "protein" + its SAXS curve).
3. **Forward-model** the scattering profile `I(q)` from the 3D coordinates with the
   **Debye formula** on both the CPU (`reference_cpu.cpp`) and the GPU (`kernels.cu`,
   one thread per `q`), and **verify** they agree (`PASS`/`FAIL`).
4. **Analyze** the curve: recover the radius of gyration `Rg` from the low-`q`
   Guinier region and report the reduced χ² of the model-vs-experiment fit.
5. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries timing and the numeric error (which vary run to run), so it is
  shown but never diffed.

## Expected result

```
2.24 -- SAXS / SANS Data-Driven Structure Modeling
Debye forward model: 40 atoms, 24 q-points (1 GPU thread per q)
normalized profile I(q)/I(0):
  q=0.0125 1/A   I/I0=1.000000
  q=0.0875 1/A   I/I0=0.618212
  q=0.1625 1/A   I/I0=0.166185
  q=0.2375 1/A   I/I0=0.032051
  q=0.3000 1/A   I/I0=0.033227
Guinier Rg (from 6 low-q pts) = 13.815 A   (synthetic true Rg = 13.671 A)
model-vs-experiment fit: scale=0.999496  reduced chi^2=0.804612
RESULT: PASS (GPU matches CPU within rel tol=1.0e-09)
```

## How to read it

- **`I(q)/I(0)`** is the normalized scattering profile. It starts at 1 (at the lowest
  `q`) and falls off — the rate of the initial fall encodes the molecule's size.
- **Guinier `Rg`** is read from the slope of `ln I` vs `q²` over the first few points
  (`ln I ≈ ln I(0) − (Rg²/3)·q²`). Recovering ≈ 13.8 Å when the structure's *true*
  `Rg` is 13.67 Å shows the pipeline measures real geometry from the curve — a
  science check, not just CPU==GPU agreement.
- **reduced χ² ≈ 0.8** (near 1) means the model curve fits the synthetic experiment
  to within its noise — exactly what we expect since the data was generated from the
  model plus 1% noise.
- **`max_rel_err ≈ 7e-16`** (on stderr) is the GPU-vs-CPU relative agreement: machine
  precision, because both call the *identical* Debye routine in `saxs_core.h`.
