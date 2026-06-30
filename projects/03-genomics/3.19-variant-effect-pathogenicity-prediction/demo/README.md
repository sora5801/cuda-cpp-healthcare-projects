# Demo — 3.19 Variant Effect / Pathogenicity Prediction

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/variants_sample.txt` (12 synthetic variants).
3. **Score** every variant with the toy CNN on both the **GPU** (`kernels.cu`) and the
   **CPU reference** (`reference_cpu.cpp`), where each variant's effect is the delta
   `score(ALT window) − score(REF window)` (in-silico mutagenesis).
4. **Verify** the GPU result against the CPU reference and print a clear `PASS`/`FAIL`.
5. **Rank** the variants and print the top-5 most "pathogenic-looking".
6. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*, not a
   benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so it is
  shown but never diffed.

## Expected result

```
3.19 -- Variant Effect / Pathogenicity Prediction
Batched in-silico mutagenesis: 12 variants, 21-base context, delta = score(ALT) - score(REF)
top-5 most pathogenic-looking variants:
  #1  pos 101096  T>G  delta = +0.321231
  #2  pos 100685  A>G  delta = +0.286602
  #3  pos 100000  T>G  delta = +0.247159
  #4  pos 100274  A>G  delta = +0.142423
  #5  pos 100411  A>C  delta = +0.012040
RESULT: PASS (GPU matches CPU within tol=1.0e-09)
```

## How to read it

- The **top 3** hits (`T>G`, `A>G`, `T>G`) are exactly the variants whose alternate
  allele completes the planted deleterious motif `CAGCT` at the centre of their window.
  The synthetic sample is engineered so the ranking recovers them (PATTERNS.md §6) —
  proof the toy model and the GPU path agree on something meaningful, not just numerically.
- `RESULT: PASS` means the GPU and CPU deltas agree within `1e-9`. They actually match to
  `~3e-17` (see the `[verify] max_abs_err` line on stderr) because both call the same
  `vep_model.h` core; the residual is pure floating-point FMA rounding.
- The stderr `[model]` banner reminds you the network is **untrained and synthetic** — no
  clinical meaning.

> **Honesty:** the data and the model are both synthetic and labelled as such. Nothing
> here is a real pathogenicity prediction.
