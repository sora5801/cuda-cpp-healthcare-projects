# Demo — 1.31 Solvent-Accessible Surface Area (SASA) on GPU

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/molecule_sample.xyz` (27 synthetic
   atoms with an engineered exposure pattern).
3. **Verify** the GPU result against the CPU reference (`reference_cpu.cpp`): the
   per-atom **exposed-point counts must match exactly** (they are integers from
   the *same* shared `__host__ __device__` function), and the derived areas agree
   to ~1e-9 Å². Prints a clear `PASS`/`FAIL`.
4. **Report** the total SASA and the **top-5 most exposed atoms**, and **time** the
   kernel (CUDA events) vs. the CPU baseline — a *teaching artifact*, not a
   benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## Expected result

See [`expected_output.txt`](expected_output.txt) for the exact stdout the demo
asserts. A green `PASS` means the GPU's exposed-point counts matched the CPU
reference **exactly**. What to notice in the output:

- the **central atom (atom[0]) is buried** — it does not appear in the most-exposed
  ranking, and its SASA is ~0 Å²;
- the **lone O/N atoms are fully exposed** (96/96 test points), which is the known
  answer baked into the synthetic geometry.

> The coordinates are **synthetic** (an engineered test geometry); the SASA values
> are a geometric exercise only and carry no chemical meaning.
