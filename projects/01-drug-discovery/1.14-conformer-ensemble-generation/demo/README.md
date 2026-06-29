# Demo — 1.14 Conformer Ensemble Generation

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/conformer_params.txt` input.
3. **Verify** the GPU energies against the CPU reference (`reference_cpu.cpp`) and print a clear `PASS`/`FAIL`.
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so it is shown but never diffed.

## What you are looking at

- `enumerated 243 conformers; pruned to 81 distinct` — the whole pipeline in one line: 3 rotamers × 5
  torsions = 243 candidate shapes, reduced by RMSD clustering (≥ 1.0 Å) to 81 non-redundant ones.
- `global minimum: conformer #0 ... torsions +180 +180 +180 +180 +180` — the **all-anti extended chain**, the
  chemically correct lowest-energy shape of a saturated chain. This is a *known-answer* sanity check on the
  embedding and the force field.
- `RESULT: PASS (GPU matches CPU within tol=1.0e-09)` — the GPU's 243 energies match the serial CPU
  reference; the stderr line shows the actual `max_abs_err` (~`5e-12` kcal/mol, pure FMA rounding).

## Expected result (stdout)

```
1.14 -- Conformer Ensemble Generation
molecule: chain of 8 atoms, 5 rotatable torsions, 3 rotamers each
enumerated 243 conformers; pruned to 81 distinct (RMSD >= 1.00 A)
global minimum: conformer #0  E = 0.105969 kcal/mol
  torsions (deg): +180 +180 +180 +180 +180
ensemble (top 5 representatives by energy):
  #1  conformer 0  E = 0.105969 kcal/mol
  #2  conformer 1  E = 0.557316 kcal/mol
  #3  conformer 2  E = 0.557316 kcal/mol
  #4  conformer 54  E = 0.561138 kcal/mol
  #5  conformer 27  E = 0.561138 kcal/mol
RESULT: PASS (GPU matches CPU within tol=1.0e-09)
```

The stderr (shown by the demo, not diffed) adds the data source, the CPU/GPU timing, and the measured
`max_abs_err`.
