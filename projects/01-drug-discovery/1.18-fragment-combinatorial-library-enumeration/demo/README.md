# Demo — 1.18 Fragment / Combinatorial Library Enumeration

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/synthons_sample.txt` (a synthetic
   3-slot × 6-block catalog → 216 products).
3. **Enumerate** every product on the GPU (one thread per product), apply the
   Lipinski + Veber drug-likeness filter, and count the passers.
4. **Verify** the GPU result against the CPU reference (`reference_cpu.cpp`) —
   counts, summed MW (fixed point), and the first-K indices must match **exactly** —
   and print a clear `RESULT: PASS`/`FAIL`.
5. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and run-varying detail, so it is shown but never
  diffed.

## Expected result

```
1.18 -- Fragment / Combinatorial Library Enumeration
combinatorial library: 6 x 6 x 6 slots = 216 products
drug-like (Lipinski+Veber) passes: 130 / 216  (60.2%)
sum of MW over passing products: 47136.000 g/mol
first 8 passing products (by index):
  #1  product[0]  = A00 + B00 + C00
  #2  product[1]  = A01 + B00 + C00
  #3  product[2]  = A02 + B00 + C00
  #4  product[3]  = A03 + B00 + C00
  #5  product[4]  = A04 + B00 + C00
  #6  product[5]  = A05 + B00 + C00
  #7  product[6]  = A00 + B01 + C00
  #8  product[7]  = A01 + B01 + C00
RESULT: PASS (GPU matches CPU exactly: count, MW-sum, indices)
```

## What to look at

- **The pass fraction (60.2 %)** is engineered into the synthetic data so the
  result is interesting and externally verifiable (not 0 % or 100 %).
- **The `PASS` line** comes from an *exact* CPU↔GPU comparison — possible because
  both sides use integer counts and a fixed-point MW sum (no floating-point
  reduction drift; see `THEORY.md` §5).
- **The first-8 indices** are the lightest corner of the library (slot 0 is the
  fastest "odometer" digit), recovered from the GPU's dense pass-flag array in
  canonical order.

> The committed data is **synthetic** and labeled as such — no chemical or
> clinical conclusion may be drawn from it.
