# Demo — 2.23 Protein-Ligand Interaction Energy Decomposition

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/complex_sample.txt` input.
3. **Verify** the GPU per-residue decomposition against the CPU reference
   (`reference_cpu.cpp`) and print a clear `PASS`/`FAIL`.
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## How to read the result

The table lists, for every protein residue, its trajectory-averaged interaction
energy with the ligand split into three physical components (kcal/mol; negative =
favourable to binding):

- **elec** — Coulomb electrostatics (salt bridges, hydrogen bonds).
- **vdw** — Lennard-Jones van der Waals (shape complementarity / packing).
- **gb** — Generalized-Born desolvation (the implicit-solvent cost of burying charge).
- **total** — the sum; the **hot-spot ranking** sorts residues by this.

Watch for the two engineered hot spots: **ARG41** (an electrostatic salt-bridge
hot spot — large favourable `elec`, partly cancelled by the `gb` desolvation
penalty) and **LEU88** (a pure van der Waals hot spot — favourable `vdw`, zero
`elec`). This is exactly the electrostatic-vs-shape distinction that guides
mutational scanning.

## Expected result

```
2.23 -- Protein-Ligand Interaction Energy Decomposition
system: 12 residues, 4 ligand atoms, 6 frames, cutoff 12.0 A
per-residue MM-GBSA decomposition (trajectory-averaged, kcal/mol):
  residue        elec        vdw         gb      total
  ARG41      -74.4941    -0.2187   +71.6892    -3.0236
  LEU88       +0.0000    -0.5403    +0.0000    -0.5403
  GLY02       +2.9920    -0.0004    -2.9546    +0.0370
  GLY03       -5.1044    -0.0013    +5.0406    -0.0652
  GLY04       -2.3061    -0.0008    +2.2773    -0.0296
  GLY05       -0.4868    -0.0002    +0.4807    -0.0063
  GLY06       +7.3709    -0.0035    -7.2787    +0.0887
  GLY07       +1.0028    -0.0071    -0.9902    +0.0055
  GLY08       +1.8133    -0.0034    -1.7906    +0.0193
  GLY09       +0.8175    -0.0015    -0.8073    +0.0087
  GLY10       -4.4759    -0.0003    +4.4200    -0.0562
  GLY11       +1.2083    -0.0025    -1.1931    +0.0126
top-3 binding hot-spot residues (most favorable total):
  #1  ARG41     total =    -3.0236 kcal/mol
  #2  LEU88     total =    -0.5403 kcal/mol
  #3  GLY03     total =    -0.0652 kcal/mol
RESULT: PASS (GPU matches CPU within tol=1.0e-04 kcal/mol)
```

The `stderr` stream additionally reports the data source, CPU/GPU timings (which
vary), and `max_abs_err` (typically ~`1e-15` kcal/mol — the CPU and GPU run the
same double-precision formula, so they agree to the last bit).

> All numbers come from a **synthetic** system (`data/README.md`). They illustrate
> the method; they are not a real binding energy and carry no clinical meaning.
