# Demo — 1.12 Molecular Fingerprint Similarity Search

## What this demonstrates

`run_demo.ps1` (Windows) / `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/fingerprints_sample.txt` (1 query vs 64 synthetic
   2048-bit library fingerprints).
3. **Verify** the GPU Tanimoto scores against the CPU reference and print a
   clear `PASS`/`FAIL`.
4. **Report** the **top-5 most similar** library fingerprints, and time the
   kernel (CUDA events) vs. the CPU baseline.

Output is split: the deterministic **top-K + PASS** goes to **stdout** (diffed
against [`expected_output.txt`](expected_output.txt)); the **timing** and the
numeric error go to **stderr** (shown, not diffed).

## Canonical output

See [`expected_output.txt`](expected_output.txt) for the exact stdout the demo
asserts. A green `PASS` means the GPU result matched the CPU reference within
`1e-6` (they agree bit-for-bit — popcount is exact integer arithmetic).

> The similarity values reflect the **synthetic** sample (engineered to span a
> wide range); they carry no chemical meaning.
