# Demo — 3.25 Base Quality Score Recalibration (BQSR)

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/bqsr_sample.txt` (1200 synthetic reads
   × 12 bp over a 24 bp reference, with 2 known-variant sites).
3. **Verify** the GPU covariate table and recalibrated qualities against the CPU
   reference (`reference_cpu.cpp`) — **exactly** (integer atomics commute), and
   print a clear `PASS`/`FAIL`.
4. **Time** the kernels (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the mismatch counts (which vary run to run in
  the timing), so it is shown but never diffed.

## Expected result

```
3.25 -- Base Quality Score Recalibration (BQSR)
alignment: 1200 reads x 12 bp, reference 24 bp, 11696 covariate bins
bases tallied = 12937 (of 14400; rest masked/skipped), observed errors = 154
per-reported-Q recalibration (aggregated over cycle & context):
  Q=30  obs= 12937  err=  154  ->  Q_emp=19
recalibrated qualities changed = 12937 / 14400 bases
RESULT: PASS (GPU table + recalibrated Q match CPU exactly)
```

**Read the headline line.** The reads were *reported* at **Q30** (a claimed 0.1%
error rate), but the empirical error rate over 12,937 tallied bases is
154/12937 ≈ 1.2%, which recalibrates to **Q_emp = 19**. That gap — reported Q30,
real ~Q19 — is exactly the systematic miscalibration BQSR exists to correct. The
two known-variant columns are **masked out**, so the genuine alternate alleles
there do not inflate the error count.

> The data is **synthetic** reads with a *known injected* error rate, not real
> sequencing — a demonstration of the covariate-table mechanics, not a clinical
> analysis. `RESULT: PASS` means the GPU and CPU produced the **same integer table
> and the same recalibrated qualities exactly** (`0 mismatches`).
