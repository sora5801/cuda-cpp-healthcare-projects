# Demo — 1.15 Protein-Ligand Binding Affinity Scoring (ML)

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/complexes_sample.txt` (6 synthetic docked
   protein-ligand poses).
3. **Verify** the GPU 3D-CNN scores against the CPU reference
   (`src/reference_cpu.cpp`) and print a clear `PASS`/`FAIL`.
4. **Report** the predicted binding affinity (pKd) for each complex and the
   **rank-1 predicted binder**, and time the kernel (CUDA events) vs. the CPU
   baseline — a *teaching artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt) — the per-complex pKd table, the
  rank-1 binder, and the `PASS` line.
- **stderr** carries the timing and the numeric verification error (which vary run
  to run), so it is shown but never diffed.

## Canonical output

See [`expected_output.txt`](expected_output.txt) for the exact stdout the demo
asserts. A green `PASS` means the GPU result matched the CPU reference within
`1e-6` pKd. The two agree to near machine precision (`~2e-15` on the reference
machine); the tolerance exists only because the GPU pools voxels with a tree
reduction while the CPU uses a flat sum — see THEORY §6.

```
1.15 -- Protein-Ligand Binding Affinity Scoring (ML)
3D-CNN scorer: 6 complexes, 16^3 grid, 8 in-ch, 8 conv filters
predicted binding affinity (pKd) per complex:
  complex  0  atoms= 35  pred_pKd= 8.1093  (synthetic_label= 5.053)
  ...
rank-1 predicted binder: complex 1  (pred_pKd=8.2245)
RESULT: PASS (GPU matches CPU within tol=1.0e-06)
```

> **Honesty note:** the network is **untrained** (deterministic stand-in weights)
> and the data is **synthetic**, so the predicted pKd values are *not* chemically
> meaningful. The demo proves the **GPU forward pass is correct** (it matches an
> independent CPU implementation), not that the affinities are real. See README
> "Limitations & honesty".
