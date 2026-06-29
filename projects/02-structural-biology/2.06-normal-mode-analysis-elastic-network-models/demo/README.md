# Demo — 2.06 Normal Mode Analysis / Elastic Network Models

## What this demonstrates

`run_demo.ps1` (Windows) / `run_demo.sh` (Linux/CMake) will:

1. **Build** the project (links **cuSOLVER**) if the executable is missing.
2. **Run** it on `data/sample/protein_ca.txt` (60 Cα atoms).
3. **Verify** that **cuSOLVER's eigenvalues match a CPU Jacobi eigensolver**.
4. **Report** the number of rigid-body modes, the lowest functional-mode
   frequencies, and per-residue mobility (predicted flexibility).

stdout (modes + mobility) is deterministic and diffed against
[`expected_output.txt`](expected_output.txt); the timing line is on stderr only.

## Canonical output

See [`expected_output.txt`](expected_output.txt). The ANM Hessian (180×180) has
exactly **6 zero (rigid-body) modes** — 3 translations + 3 rotations — as it must;
the next eigenvalues are the slow, large-scale functional motions. `RESULT: PASS`
means cuSOLVER's eigenvalues agree with the CPU Jacobi reference (here to ~`1e-12`,
machine precision). The per-residue mobility highlights the most flexible
residues. cuSOLVER edges out the CPU even on this small matrix; the gap explodes
for real proteins (the eigensolver is O(n³), n = 3N).

> The structure is **synthetic** — a demonstration of NMA + the cuSOLVER
> eigensolver, not a real protein analysis.
