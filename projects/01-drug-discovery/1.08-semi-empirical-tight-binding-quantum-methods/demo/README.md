# Demo — 1.8 Semi-Empirical & Tight-Binding Quantum Methods

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/molecules_sample.txt` batch of eight conjugated molecules.
3. For each molecule, **build a Hückel/tight-binding Hamiltonian, diagonalise it on the GPU** (cuSOLVER
   batched Jacobi), fill the π electrons, and report total π-energy, HOMO, LUMO, and the **HOMO–LUMO gap**.
4. **Verify** the GPU eigenvalues against the CPU Jacobi reference (`reference_cpu.cpp`) and print a clear
   `PASS`/`FAIL`.
5. **Time** the build kernel and the batched solve (CUDA events) plus the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against [`expected_output.txt`](expected_output.txt).
- **stderr** carries timing and the numeric error (which vary run to run), so it is shown but never diffed.

## What to look for

- The energies are **exactly the textbook Hückel values** (a built-in analytic check): benzene
  `E_π = −8.000 |β|` with gap `2.000`, butadiene `−4.472`, naphthalene `−13.683`.
- **Cyclobutadiene has a HOMO–LUMO gap of exactly 0** — it is *antiaromatic* and flagged as the most
  reactive molecule in the batch. That zero gap (a pair of degenerate non-bonding orbitals) is the single
  most interesting number to stare at; contrast it with benzene's large, stable gap of 2.
- `RESULT: PASS` means the GPU batched eigensolve agrees with the CPU reference within `1.0e-09`
  (the actual worst difference, on stderr, is `~3e-15` — machine precision).

## Expected result (stdout)

```
1.8 -- Semi-Empirical & Tight-Binding Quantum Methods
Huckel tight-binding on 8 molecules (padded dim N=10)
energies in units of |beta| (alpha=0, beta=-1)
molecule         atoms           E_pi       HOMO       LUMO        gap
ethylene             2      -2.000000  -1.000000   1.000000   2.000000
allyl                3      -2.828427   0.000000   1.414214   1.414214
butadiene            4      -4.472136  -0.618034   0.618034   1.236068
benzene              6      -8.000000  -1.000000   1.000000   2.000000
cyclobutadiene       4      -4.000000   0.000000   0.000000   0.000000
hexatriene           6      -6.987918  -0.445042   0.445042   0.890084
cyclopentadienyl     5      -5.854102  -0.618034   1.618034   2.236068
naphthalene         10     -13.683239  -0.618034   0.618034   1.236068
smallest HOMO-LUMO gap: cyclobutadiene (gap=0.000000 |beta|) -- most reactive/polarizable
RESULT: PASS (GPU batched eigensolve matches CPU Jacobi within tol=1.0e-09)
```

The stderr lines (timing, data source, worst eigenvalue difference) will differ between machines/runs;
that is expected and is why they are not part of the diff.
