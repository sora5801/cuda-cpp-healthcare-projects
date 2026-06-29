# Demo — 1.3 Molecular Docking Engine

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/receptor_ligand_sample.txt` — a tiny
   **synthetic** docking problem: a Gaussian energy *well* (one attractive pocket)
   plus a small rigid 5-atom ligand, with 9261 candidate poses to score.
3. **Verify** the GPU result against the CPU reference (`reference_cpu.cpp`): they
   must pick the **same winning pose index** and agree on its energy, printing a
   clear `PASS`/`FAIL`.
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the index/energy diagnostics (which vary run
  to run), so it is shown but never diffed.

## Expected result (stdout)

```
1.3 -- Molecular Docking Engine
Rigid-body docking: scored 9261 poses (7^3 translations x 3^3 rotations) over a 16x16x16 energy grid
ligand atoms: 5
best pose index: 4967
best pose translation (A): tx=0.5000 ty=-0.5000 tz=0.0000
best pose rotation (rad):  a=4.1888 b=2.0944 c=2.0944
best score (kcal/mol): -41.367550
RESULT: PASS (GPU best pose matches CPU; energies agree within tol=1.0e-09)
```

## How to read it

The synthetic well sits at world coordinate **(0.5, -0.5, 0.0) Å**. The engine's
best translation is **tx=0.5, ty=-0.5, tz=0.0** — it recovers the well's location
exactly (the ligand centroid drops straight into the pocket). That the answer is
*known by construction* is what makes this demo verifiable by eye
(`scripts/make_synthetic.py` builds the well; see `data/README.md`).

A representative timing line on stderr (an RTX 2080, `sm_75`) looks like:

```
[timing] CPU reference: 1.6 ms   GPU kernel: 0.7 ms
[verify] index match: yes   CPU idx=4967  GPU idx=4967   energy_err=0.000e+00 (tol 1.0e-09)
```

`energy_err = 0` because the CPU and GPU evaluate the **same** `score_pose()` on the
**same** winning pose (`docking_core.h`, the shared `__host__ __device__` core). The
timing is illustrative only: at this toy size the GPU is launch/copy-bound; its
advantage grows with the number of poses and ligands (see `THEORY.md`).
