# Demo — 1.22 Constant-pH Molecular Dynamics

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed synthetic system `data/sample/cph_system.txt`.
3. **Verify** the GPU titration tally against the CPU reference
   (`reference_cpu.cpp`) — an **exact integer match** — and print `PASS`/`FAIL`.
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately (PATTERNS.md §3):

- **stdout** is byte-for-byte deterministic — the titration curves (fraction
  protonated per residue, as integer percent) and the predicted pKa values — and
  is diffed against [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the verification counts (which vary run to
  run), so it is shown but never diffed.

## How to read the output

- The **fraction-protonated table** is the titration curve: each row is a
  residue, each column a pH. As pH rises, every residue eventually deprotonates,
  so the numbers fall from ~100% to ~0%.
- The **predicted pKa** of a residue is the pH where its curve crosses 50%
  (linear-interpolated). The `shift` column is `predicted − intrinsic`: the
  electrostatic coupling between residues moves the apparent pKa.
  - **ASP** (acid) shifts **down** (~−1.9): nearby cations stabilise its
    deprotonated negative form.
  - **LYS** (base) shifts **up** (~+0.7): the nearby ASP anion stabilises its
    protonated positive form.
  - **HIS** (in the middle) barely shifts: it feels opposing effects from each side.

That coupling-driven pKa shift is the core lesson of constant-pH simulation.

## Expected result

```
1.22 -- Constant-pH Molecular Dynamics
reduced-scope teaching model: ensemble Metropolis MC titration
residues = 3, pH grid = 0.0..14.0 in 15 steps, replicas = 8
coupling k = 12.0 kcal*A/mol/e^2, kT = 0.593 kcal/mol, sweeps = 6000 (burn-in 1000)

fraction protonated (%), rows = residue, cols = pH:
         pH 0.0 pH 1.0 pH 2.0 pH 3.0 pH 4.0 pH 5.0 pH 6.0 pH 7.0 pH 8.0 pH 9.0 pH10.0 pH11.0 pH12.0 pH13.0 pH14.0
ASP         99    93    57    11     1     0     0     0     0     0     0     0     0     0     0
HIS        100   100   100   100   100    97    76    24     3     0     0     0     0     0     0
LYS        100   100   100   100   100   100   100   100   100    99    93    57    12     1     0

predicted pKa (curve crosses 50%) vs intrinsic:
  ASP  intrinsic  4.00  ->  pKa  2.15  (shift -1.85)
  HIS  intrinsic  6.50  ->  pKa  6.50  (shift -0.00)
  LYS  intrinsic 10.50  ->  pKa 11.16  (shift +0.66)

RESULT: PASS (GPU protonation tally matches CPU exactly)
```

The exact integers above come from the committed seed (`20260628`) and the shared
deterministic RNG, so a correct build reproduces them byte-for-byte on any GPU.
