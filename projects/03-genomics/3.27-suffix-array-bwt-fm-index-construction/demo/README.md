# Demo — 3.27 Suffix Array / BWT / FM-Index Construction

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/dna_sample.txt` input.
3. **Verify** the GPU suffix array against the CPU reference
   (`src/reference_cpu.cpp`) — exact integer match — plus BWT and FM-count
   agreement, and print a clear `PASS`/`FAIL`.
4. **Time** the GPU doubling kernels (CUDA events) and the CPU baseline — a
   *teaching artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and run-varying detail, so it is shown but never
  diffed.

## Expected result

```
3.27 -- Suffix Array / BWT / FM-Index Construction
text length (with $ sentinel): 61
suffix array SA[0:12] = 60 9 0 20 40 10 30 50 38 36 18 26
BWT[0:32] = GT$TGACGGCGTTCTTGCCGTAAAAAATCAGA
FM-index backward search: pattern "ACG" occurs 6 time(s)
verify: SA mismatches=0  BWT match=yes  FM match=yes
RESULT: PASS (GPU suffix array matches CPU exactly, tol=0)
```

## How to read it

- **`SA[0:12]`** — the first twelve entries of the suffix array (the starting
  positions of the 12 lexicographically smallest suffixes). `SA[0] = 60` is the
  `$` sentinel suffix, which always sorts first.
- **`BWT[0:32]`** — the first 32 characters of the Burrows-Wheeler transform (the
  "last column"). Note how it clusters repeated contexts (`AAAAAA`), which is why
  the BWT compresses well.
- **`"ACG" occurs 6 time(s)`** — the FM-index backward search recovers the planted
  motif count (the synthetic input plants `ACGT` repeatedly; see
  [`../data/README.md`](../data/README.md)).
- **`SA mismatches=0  BWT match=yes  FM match=yes`** — the independent CPU and GPU
  implementations agree on every value; since the suffix array is an integer
  permutation, the only acceptable result is an exact match (tolerance `0`).

The `[timing]` lines on stderr will vary; at this tiny size the GPU is
launch-bound and slower than the CPU — expected, and explained in
[`../THEORY.md`](../THEORY.md) §4 and §7.
