# Demo — 1.4 Ultra-Large Virtual Screening

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/ligands_sample.txt` (1 target vs 64 synthetic
   ligands).
3. **Verify** the GPU per-ligand scores against the CPU reference
   (`reference_cpu.cpp`) and print a clear `PASS`/`FAIL`.
4. **Report** how many ligands passed the drug-likeness cascade and the
   **top-5 hits** by surrogate score, and time the kernel (CUDA events) vs. the
   CPU baseline — a *teaching artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the mismatch count (which vary run to run, or
  are run-context detail), so it is shown but never diffed.

## Expected result

See [`expected_output.txt`](expected_output.txt) for the exact stdout the demo
asserts. A green `PASS` means the GPU result matched the CPU reference **exactly**
(`mismatches = 0` — the scores are integer fixed-point, so the two agree
bit-for-bit, tolerance zero).

The four top hits all carry feature mask `0x0000A5B3` (the target's required
features) and sit on the target's ideal size/logP — these are the **designed
binders** the synthetic sample plants, so seeing them at the top validates the
science, not just CPU==GPU agreement.

> The descriptor values and scores reflect the **synthetic** sample; they carry
> no chemical meaning.
