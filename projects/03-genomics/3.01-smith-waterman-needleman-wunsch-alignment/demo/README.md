# Demo — 3.01 Smith-Waterman / Needleman-Wunsch Alignment

## What this demonstrates

`run_demo.ps1` (Windows) / `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/sequences_sample.txt` (two synthetic DNA sequences
   that share a mutated motif).
3. **Verify** that the GPU's anti-diagonal wavefront fills the **exact same DP
   matrix** as the serial CPU reference (`PASS` iff every cell matches).
4. **Report** the best local-alignment **score**, the endpoint cell, percent
   identity, and a preview of the aligned columns. Timing goes to stderr.

stdout (score + alignment) is deterministic and diffed against
[`expected_output.txt`](expected_output.txt); the timing line varies and is shown
on stderr only.

## Canonical output

See [`expected_output.txt`](expected_output.txt). A green `PASS` means the GPU
wavefront and the CPU DP produced an identical score matrix.

> **Honesty:** for a single small alignment the per-diagonal launches make the
> GPU slower than the CPU — that is expected and explained in `THEORY.md`. The
> wavefront pays off on large matrices and batched alignments. The numbers are a
> teaching artifact, not a benchmark.
