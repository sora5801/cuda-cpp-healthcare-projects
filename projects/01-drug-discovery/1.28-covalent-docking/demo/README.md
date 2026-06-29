# Demo — 1.28 Covalent Docking

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/covalent_sample.txt` input.
3. **Verify** the GPU result against the CPU reference (`reference_cpu.cpp`) and
   print a clear `PASS`/`FAIL`.
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program scores all `36³ = 46 656` ligand torsion conformations — one GPU
thread each — and reports the lowest-energy (best) docked pose. The CPU does the
same serial scan; both then `argmin` to the same pose.

The output is split deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## Expected result

```
1.28 -- Covalent Docking
covalent docking: 46656 conformations (3 torsions x 36 samples)
best pose: id=46401  energy=-2.347011 kcal/mol
best torsions (deg): 330.0 280.0 350.0
warhead-Sgamma bond = 1.810 A (ideal 1.810)
ligand atom[0] = (-0.500, 1.414, 0.000) A
RESULT: PASS (GPU matches CPU within tol=1.0e-06)
```

### How to read it

- **best pose: id / energy** — the winning conformation's flat index and its total
  energy (covalent constraint + ligand–pocket interaction). Lower is better; the
  negative value means a favorable fit.
- **best torsions (deg)** — the three rotatable-bond angles of the docked pose.
- **warhead-Sgamma bond** — the covalent bond length actually achieved vs the
  ideal 1.81 Å (a chemistry sanity check; here it sits exactly at the ideal).
- **ligand atom[0]** — coordinates of the first ligand atom of the best pose.
- **RESULT: PASS** — the GPU energy array matched the CPU array to within
  `1e-6 kcal/mol` (the stderr line shows the actual error, ~`2e-15`, i.e. machine
  precision: the CPU and GPU run the *same* double-precision math).

The timing line on **stderr** (e.g. `CPU 15 ms / GPU 1 ms`) is illustrative only —
the GPU's edge grows as the number of torsions (and thus conformations) rises.

> **Not for clinical use.** The geometry, force field, and pocket are synthetic
> and didactic.
