# Demo — 3.16 Sequence Error Correction

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/reads_sample.txt` (120 synthetic
   reads carrying ~2% substitution errors, plus the error-free truth).
3. **Build the k-mer spectrum** (phase 1) and **correct the reads** (phase 2) on
   both the GPU and a CPU reference, then **verify** they are *byte-identical*
   (the spectrum tables and the corrected reads must match exactly) — printed as
   `PASS`/`FAIL`.
4. Report the **science metric**: how many erroneous bases existed *before* vs
   *after* correction (so you can see the method actually works), and **time**
   each kernel with CUDA events — a *teaching artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the per-run timings (which vary), so it is shown but never
  diffed.

## Expected result

```
3.16 -- Sequence Error Correction
k-mer spectrum error correction (k=9, trust T=3)
reads: 120   total bases: 7200
spectrum: 1247 distinct 9-mers observed
corrections applied: 99 base(s) over 120 read(s)
errors vs truth:  before = 132   after = 39   (removed 93)
verify: spectrum_mismatch=0  corrected_mismatch=0
RESULT: PASS (GPU matches CPU exactly: spectrum + corrected reads)
```

## How to read it

- **spectrum: 1247 distinct 9-mers** — the count table found 1247 different
  9-mers across all reads. The true genome contributes the recurring (trusted)
  ones; sequencing errors add a long tail of rare (untrusted) ones.
- **corrections applied: 99** — the corrector substituted 99 bases that were
  covered only by untrusted k-mers but became trusted after a single flip.
- **errors before = 132, after = 39 (removed 93)** — measured against the known
  truth: correction removed ~70% of the substitution errors. The 39 residual
  errors are honest (errors near read ends have no trusted k-mer to anchor on,
  and clustered errors can defeat a single-pass corrector) — see
  [README "Limitations"](../README.md#limitations--honesty).
- **spectrum_mismatch = 0, corrected_mismatch = 0** — the GPU and CPU produced
  identical results, which is the correctness guarantee. Because every operation
  is integer/byte work, agreement is **exact** (no floating-point tolerance).
