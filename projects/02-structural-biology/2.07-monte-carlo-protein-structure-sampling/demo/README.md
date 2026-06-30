# Demo — 2.7 Monte Carlo Protein Structure Sampling

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/hp_problem.txt` synthetic HP chain.
3. **Verify** the GPU result against the CPU reference (`reference_cpu.cpp`):
   every replica's `{best, final}` energy must match **exactly** (integer
   energies), printing a clear `PASS`/`FAIL`.
4. **Time** the GPU vs CPU Monte Carlo — a *teaching artifact*, not a benchmark.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the mismatch count (which vary run to run),
  so it is shown but never diffed.

## What you are looking at

- `best energy found = -8 (8 H-H contacts)` — the lowest energy any of the 256
  replicas reached. Energy is `E = -(number of non-bonded H–H contacts)`, so a
  more negative number is a better-folded (more compact, more "buried") state.
  Recovering 8 H–H contacts from a straight starting chain shows the ensemble
  genuinely *folds* the synthetic sequence — the hydrophobic collapse that drives
  real protein folding, captured in a toy lattice model.
- `ensemble mean best energy = -1114/256` — a deterministic ensemble statistic,
  printed as an exact integer fraction (sum over replicas / replica count) so it
  stays reproducible (no float rounding).
- `RESULT: PASS` — the GPU's 256 parallel walks reproduced the CPU's 256 serial
  walks *exactly*, because both share the RNG and the Boltzmann tables.

## Expected result (stdout)

```
2.7 -- Monte Carlo Protein Structure Sampling
[reduced-scope teaching model: 2-D HP lattice protein, Metropolis MC]
sequence (n=18, 10 H): HPHPPHHPHHPHHPPHPH
replicas = 256, sweeps = 600, T in [0.30, 3.00]
best energy found = -8 (8 H-H contacts) by replica 15
ensemble mean best energy = -1114/256
RESULT: PASS (GPU per-replica energies match CPU exactly)
```

A representative **stderr** line (varies, not diffed):

```
[timing] CPU MC: 131.221 ms   GPU MC: 107.954 ms
```

The exact ms numbers change every run; the GPU's edge grows as you add replicas
(`make_synthetic.py --replicas 1024`). The point is *correctness you can see*,
not the speed-up magnitude.
