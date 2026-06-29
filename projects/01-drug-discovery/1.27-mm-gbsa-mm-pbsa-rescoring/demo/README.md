# Demo — 1.27 MM-GBSA / MM-PBSA Rescoring

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/complex_sample.txt` input — a tiny,
   **synthetic** protein–ligand complex with a 6-frame "trajectory".
3. **Rescore** every snapshot on the GPU (one thread per snapshot, calling the
   shared `snapshot_dg()` physics) and on the CPU reference, then **verify** the
   GPU result against the CPU one and print a clear `PASS`/`FAIL`.
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## Expected result

```
1.27 -- MM-GBSA / MM-PBSA Rescoring
Rescoring 1 complex: receptor=3 atoms, ligand=2 atoms, snapshots=6
per-snapshot dG (kcal/mol):
  frame  0 : dG =     7.2021
  frame  1 : dG =     7.6902
  frame  2 : dG =     7.8888
  frame  3 : dG =     7.9382
  frame  4 : dG =     7.9553
  frame  5 : dG =     7.9640
MM-GBSA dG_bind (ensemble mean) = 7.7731 kcal/mol
RESULT: PASS (GPU matches CPU within tol=1.0e-06)
```

## How to read it

Each `frame` line is the per-snapshot binding free-energy estimate ΔG
(`E_vdw + E_elec + ΔG_GB + (−T·ΔS)`), in kcal/mol. In this **synthetic** sample
the ligand drifts out of the pocket frame by frame, so the favorable interaction
weakens and ΔG **climbs toward the bare entropy penalty** (`−T·ΔS = 8.0`): the
interaction part (`ΔG − 8.0`) goes from about **−0.80** (frame 0, most bound) up
toward **0** (frame 5, unbound). That monotone trend is a built-in sanity check
on the physics — see `docs/PATTERNS.md §6`.

`MM-GBSA dG_bind` is the **ensemble mean** over the six snapshots — exactly how a
real MM-GBSA run reports a single binding free energy from a trajectory. The
value here is **not** a real affinity; the input is synthetic.

The stderr line `max_abs_err = 0.000e+00 kcal/mol` shows the GPU and CPU agreed
to the last bit on this sample (they run the identical `snapshot_dg()` source);
the tolerance is `1e-6` to allow for a possible ~1-ULP `exp()` difference on
other inputs (see `../THEORY.md`).
