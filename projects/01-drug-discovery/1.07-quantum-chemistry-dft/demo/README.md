# Demo — 1.7 Quantum Chemistry / DFT (reduced-scope RHF/SCF)

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/h2.txt` (the hydrogen molecule).
3. **Verify** twice:
   - the **GPU two-electron integral tensor** against the CPU reference
     (`reference_cpu.cpp`) — to ~machine precision, and
   - the **final SCF total energy** computed with the cuSOLVER eigensolver
     against the CPU-Jacobi energy.
   Then print a clear `PASS`/`FAIL`.
4. **Time** the ERI kernel (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timings and the numeric verification errors (which vary
  run to run), so it is shown but never diffed.

## Expected result

```
1.7 -- Quantum Chemistry / DFT (reduced-scope RHF/SCF)
molecule: 2 atoms, 2 electrons, basis STO-3G (N=2 functions)
SCF converged in 2 iterations
nuclear repulsion :   0.71428571 Ha
electronic energy :  -1.83100004 Ha
TOTAL ENERGY      :  -1.11671432 Ha
orbital energies (Ha):  -0.57820   0.67027
HOMO =  -0.57820 Ha   LUMO =   0.67027 Ha   gap =   1.24847 Ha
ERI verify (GPU vs CPU): PASS   energy verify (GPU vs CPU): PASS
RESULT: PASS
```

## How to read it

- **TOTAL ENERGY = −1.11671432 Ha** is the famous textbook value for H₂ in the
  STO-3G basis (Szabo & Ostlund). That the program lands on it from first
  principles — only nuclear charges and positions in — is the whole point.
- **nuclear repulsion = 1/1.4 = 0.714 Ha** is just the two protons' Coulomb energy.
- The **orbital energies** are the molecular-orbital levels: the lower one
  (−0.578 Ha, the **HOMO**) is the bonding σ orbital that holds both electrons; the
  upper (+0.670 Ha, the **LUMO**) is the empty antibonding σ*.
- Both `ERI verify` and `energy verify` say `PASS`, so the GPU integral kernel and
  the cuSOLVER-driven SCF agree with the transparent CPU reference.

The `[timing]` lines on stderr show the GPU is *launch-bound* at this tiny size
(N = 2 basis functions, only 16 integrals) — see `../THEORY.md` on why the GPU's
advantage is an **O(N⁴)** story that only shows up for real-sized molecules.
