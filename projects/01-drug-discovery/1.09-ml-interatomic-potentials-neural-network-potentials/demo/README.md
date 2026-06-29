# Demo — 1.9 ML Interatomic Potentials (Neural Network Potentials)

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing (`Release|x64`).
2. **Run** it on the committed synthetic cluster
   (`data/sample/water_cluster.xyzc`, 24 atoms).
3. **Verify** the GPU per-atom energies and total energy against the CPU
   reference (`reference_cpu.cpp`) and print a clear `PASS`/`FAIL`. Both paths call
   the **same** `__host__ __device__` physics in `src/nnp.h`, so agreement is to
   floating-point round-off (~1e-15 here), not approximate.
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt). Energies are printed at fixed
  precision so the text never wobbles, and the total is summed on the host in a
  fixed atom order so it is reproducible.
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## What you are looking at

Each atom gets a **per-atom energy** `E_i` produced by: (1) eight radial
*atom-centered symmetry functions* that fingerprint its local environment, then
(2) a small MLP (8→16→16→1) that maps that fingerprint to an energy. The **total
energy** is the sum over atoms. This is the Behler-Parrinello recipe used by ANI
and friends — here in a deliberately reduced teaching form (one element,
radial-only descriptors, fixed/manufactured weights). See `../THEORY.md`.

## Expected result

```
1.9 -- ML Interatomic Potentials (Neural Network Potentials)
Behler-Parrinello NNP: 24 atoms, 8 radial descriptors, MLP 8->16->16->1
per-atom energy E_i (GPU), first 6 atom(s):
  atom  0  E = +0.880908
  atom  1  E = +0.697670
  atom  2  E = +0.722885
  atom  3  E = +0.863988
  atom  4  E = +0.724925
  atom  5  E = +0.706800
total energy E = +17.647320
RESULT: PASS (GPU matches CPU within tol=1.0e-09)
```

The energies are in the model's arbitrary (synthetic) units; their **values are
not physically meaningful** — what is real is that the GPU and CPU agree and that
the descriptor → per-atom-MLP → sum pipeline runs end to end. A typical run shows
`max per-atom err ~ 1e-15` on stderr, comfortably under the `1e-9` tolerance.
