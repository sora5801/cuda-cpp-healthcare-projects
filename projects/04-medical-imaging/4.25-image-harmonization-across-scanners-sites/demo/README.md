# Demo — 4.25 Image Harmonization Across Scanners/Sites

## What this demonstrates

Running `run_demo.ps1` (Windows / MSBuild) or `run_demo.sh` (Linux / CMake) will:

1. **Build** the project if the `Release|x64` executable is missing.
2. **Run** it on the committed synthetic table `data/sample/harmonization_sample.txt`.
3. Harmonize the table **twice** — once with the CPU reference, once on the GPU — and
   **verify** the two harmonized tables agree to ~machine precision.
4. **Diff** the deterministic stdout against [`expected_output.txt`](expected_output.txt)
   and report **PASS/FAIL**. Timings go to stderr (shown, not diffed).

## What you are looking at

```
4.25 -- Image Harmonization Across Scanners/Sites
ComBat: 24 samples x 12 features, 3 scanners, 1 covariate(s)
max across-scanner feature-mean gap:
  before harmonization = 7.738827        <- the scanner signature (large)
  after  harmonization = 0.797558        <- collapsed after ComBat
feature 0, first sample of each scanner (harmonized):
  scanner 0, sample  0: 55.821820
  scanner 1, sample  1: 55.061502
  scanner 2, sample  2: 57.121964
RESULT: PASS (GPU harmonized table matches CPU reference)
```

- **The headline result** is the *before vs after* mean gap. Before harmonization the three
  scanners disagree on a feature's mean by up to `7.74`; after ComBat that gap collapses to
  `0.80`. The scanner location/scale signature has been removed. (It is deliberately not
  exactly zero — empirical-Bayes shrinkage keeps a little for robustness; see THEORY §How
  we verify correctness.)
- **`RESULT: PASS`** means the GPU and CPU harmonized tables agree. The stderr line
  `max |GPU - CPU| = 7.105e-15` shows the actual agreement (tolerance `1e-9`); both sides
  run the same `__host__ __device__` core (`src/combat.h`), so they match to FP64 rounding.

## Why stdout is safe to diff

All run-to-run varying numbers (wall-clock timings) are written to **stderr**; **stdout**
carries only deterministic results. ComBat here uses no atomics and no cross-thread
reduction, so the harmonized values are bit-stable across runs — `expected_output.txt` was
captured from a real run and matches every time.

## Run it

```powershell
# Windows (PowerShell), from the project folder:
./demo/run_demo.ps1
```

```bash
# Linux/macOS (needs CMake + CUDA toolkit):
./demo/run_demo.sh
```
