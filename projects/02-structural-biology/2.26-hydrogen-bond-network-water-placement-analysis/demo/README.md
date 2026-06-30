# Demo — 2.26 Hydrogen Bond Network & Water Placement Analysis

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed synthetic `data/sample/water_sample.txt`.
3. **Verify** the GPU GIST tallies against the CPU reference
   (`reference_cpu.cpp`) — occupancy and fixed-point energy must match **exactly**,
   and the ranked hydration-site list must be identical — printing `PASS`/`FAIL`.
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt). Determinism comes from accumulating
  the energy in **fixed-point integers**, whose atomic adds commute.
- **stderr** carries the timing and verification detail (which vary run to run), so
  it is shown but never diffed.

## What the learner is seeing

The table ranks **hydration sites** — voxels where water clusters persistently —
and annotates each with its GIST thermodynamics:

| Column | Meaning |
|---|---|
| `n` | occupancy: water observations summed into this voxel over all frames |
| `g` | number density relative to bulk water (`1` = bulk; `≫1` = ordered site) |
| `dE` | mean water–solute energy minus the bulk reference (kcal/mol; `>0` = strained) |
| `-TdS` | translational entropy penalty `k_B T ln g` (kcal/mol; `≥0`) |
| `dG` | GIST free energy `dE − TdS` (kcal/mol); **high = displaceable** |

The two synthetic ordered waters at `(4,5,5)` and `(6,4,4)` — occupied every frame
(`g ≈ 240`) and energetically strained — top the list. Displacing such waters with
a ligand atom is the GIST/WaterMap recipe for an affinity gain (educational only).

## Expected result

```
2.26 -- Hydrogen Bond Network & Water Placement Analysis
GIST grid: 10x10x10 voxels @ 0.50 A spacing  (1000 voxels)
samples: 120 frames x 8 waters = 960 water observations; 14 solute atoms
hydration sites (voxels with adequate occupancy): 38

top 8 hydration sites (ranked by occupancy; GIST dG = displaceability, kcal/mol):
  rank  voxel(ix,iy,iz)   n      g      dE     -TdS      dG
    1   ( 4, 5, 5)        120  239.52   15.16    3.25   18.41
    2   ( 6, 4, 4)        120  239.52   14.93    3.25   18.18
    3   ( 9, 8, 0)         37  73.85   14.78    2.55   17.33
    4   ( 0, 1, 9)         35  69.86    8.97    2.52   11.48
    5   ( 0, 9, 9)         34  67.86   11.59    2.50   14.09
    6   ( 0, 9, 8)         31  61.88   13.79    2.44   16.23
    7   ( 0, 8, 9)         30  59.88   13.85    2.42   16.28
    8   ( 0, 9, 1)         29  57.88    9.00    2.40   11.41

RESULT: PASS (GPU voxel tallies + site ranking match CPU exactly)
```

The `RESULT: PASS` line and the exit code (`0`) are what `run_demo` checks. The
timing line on stderr will differ on your machine — that is expected and not part
of the comparison.
