# Demo — 3.21 Structural Variant (SV) Calling

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed synthetic sample (`data/sample/sv_sample.txt`).
3. **Call** structural variants two ways — a serial CPU reference and the GPU
   pipeline (one thread per read realigns its breakpoint by banded Smith-Waterman,
   then votes into a shared histogram with integer atomics).
4. **Verify** that the GPU histogram and the emitted SV calls match the CPU
   **exactly** (integer atomics commute → no tolerance needed) and print `PASS`/`FAIL`.
5. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and bin-mismatch detail (which vary run to run),
  so it is shown but never diffed.

## Expected result

```
3.21 -- Structural Variant (SV) Calling
reduced-scope teaching version: deletion calling by split-read
realignment (banded SW) + breakpoint clustering on SYNTHETIC data
reference length = 240 bp, candidate reads = 24, min support = 3
SV calls (sorted by breakpoint): 1
  DEL  bp=120  len=50  support=18  GT=1/1
planted truth: bp=120 len=50  -> recovered: YES
RESULT: PASS (GPU histogram+calls match CPU exactly)
```

**How to read it:** 18 of the 24 candidate reads carry the true breakpoint flank;
banded SW pulls each one's jittered position back to reference coordinate 120, so
they cluster into a single deletion call (`DEL bp=120 len=50`) with 18 supporting
reads. The 6 noise reads scatter below the `min support = 3` floor and are not
called. `recovered: YES` confirms the call matches the planted ground truth. The
genotype `1/1` follows the integer VAF rule in `src/sv.h` (support/total = 18/24
= 0.75 ≥ 3/4 → homozygous-alt in this synthetic mix). `GPU histogram+calls match
CPU exactly` is the correctness gate — every breakpoint bin agreed bit-for-bit.
