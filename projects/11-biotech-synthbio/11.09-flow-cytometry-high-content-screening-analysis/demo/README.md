# Demo — 11.09 Flow Cytometry & High-Content Screening Analysis

## What this demonstrates

`run_demo.ps1` (Windows) / `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/cytometry_sample.txt` (20,000 events × 5 markers).
3. **Verify** that the GPU k-means matches the CPU reference — **labels and
   centroids identical** (the fixed-point atomic reduction commutes).
4. **Report** each recovered cluster's size and centroid, and the inertia.

stdout (the clusters) is deterministic and diffed against
[`expected_output.txt`](expected_output.txt); the timing line is on stderr only.

## Canonical output

See [`expected_output.txt`](expected_output.txt). k-means recovers all 5 synthetic
populations with the right sizes (6000/5000/4000/3000/2000) and centroids matching
the true marker patterns. `RESULT: PASS` means the GPU and CPU produced the **same
labels and centroids exactly** (`0 label mismatches`, `centroid diff 0`). The GPU
clusters the 20k events several times faster than the CPU; the gap grows toward the
10⁶–10⁷ cells of real runs.

> The data is **synthetic** Gaussian populations, not real immunophenotyping — a
> demonstration of GPU clustering, not a clinical analysis.
