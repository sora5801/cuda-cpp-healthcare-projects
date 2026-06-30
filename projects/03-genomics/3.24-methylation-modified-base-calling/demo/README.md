# Demo — 3.24 Methylation / Modified-Base Calling

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/methylation_sample.txt` input.
3. **Verify** the per-job log-likelihood ratios computed on the GPU
   (`kernels.cu`) against the CPU reference (`reference_cpu.cpp`) and print a clear
   `PASS`/`FAIL`.
4. **Report** the per-site 5mC calls and how many match the planted ground truth.
5. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately (PATTERNS.md §3):

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## How to read the result

Each row is one CpG site:

```
  site  ref_pos  mean_LLR   call    truth
    0     1000    +15.002   5mC     5mC
    2     1274    -12.921   C       C
```

- `mean_LLR` is the average log-likelihood ratio over the 8 reads covering the
  site: `logL(methylated model) − logL(canonical model)`.
- **Positive ⇒ call 5mC**, negative ⇒ call canonical C. The clean separation
  (≈ +15 for methylated, ≈ −15 for canonical) is by design in the synthetic data —
  real signal is noisier.
- `truth` is the planted label; `calls matching ground truth: 12 of 12` confirms a
  correct DP + LLR recovers what the generator built.

## Expected result

The full deterministic stdout is captured in
[`expected_output.txt`](expected_output.txt). It ends with:

```
calls matching ground truth: 12 of 12
RESULT: PASS (GPU matches CPU within tol=1.0e-03)
```

The `PASS` means the GPU reproduced every per-job LLR within `1.0e-3` of the CPU
reference (the measured error is `0.0`, because both call the same shared
`__host__ __device__` DP core in [`../src/meth_core.h`](../src/meth_core.h)).
