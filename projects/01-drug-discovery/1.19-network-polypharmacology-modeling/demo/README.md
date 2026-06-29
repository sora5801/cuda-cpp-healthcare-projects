# Demo — 1.19 Network / Polypharmacology Modeling

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed synthetic knowledge graph in `data/sample/`.
3. **Score** the query drug against all 64 candidate protein tails with TransE
   (`h + r ≈ tail`) on the GPU, **verify** that the GPU scores match the CPU
   reference (`reference_cpu.cpp`) within tolerance, and print a clear `PASS`/`FAIL`.
4. **Rank** the candidates and report the **top-5 predicted targets** plus a
   **recovery** metric: how many of the synthetic ground-truth targets landed in
   the top-5. This is the project's self-check that the method found the answer we
   embedded in the data (`data/README.md`).
5. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric verification error (which vary run
  to run), so it is shown but never diffed.

## Expected result

```
1.19 -- Network / Polypharmacology Modeling
TransE link prediction: 1 query drug vs 64 protein tails (dim=16)
top-5 predicted targets (protein index : TransE score):
  #1  protein[6]  score = -0.000000
  #2  protein[15]  score = -0.002517
  #3  protein[62]  score = -0.008118
  #4  protein[30]  score = -4.857530
  #5  protein[52]  score = -5.265568
recovery: 3 / 3 ground-truth targets in top-5
RESULT: PASS (GPU matches CPU within tol=1.0e-05)
```

The top three predicted targets are exactly the three ground-truth targets
(`6`, `15`, `62`), each at a near-zero TransE distance (highest score), while the
random decoys sit far below. The score for `protein[6]` is `-0.000000` because the
synthetic drug + relation were constructed so `h + r` lands exactly on that
protein's embedding.

## A note on the tolerance

The verification tolerance is `1.0e-5`, not `0`. The CPU reference and the GPU
kernel call the **same** scoring function (`src/transe.h`), but `nvcc` fuses the
device-side `acc + diff*diff` into a single fused-multiply-add (FMA) while the host
compiler does a separate multiply then add. FMA keeps more intermediate precision,
so the two sums diverge by about `1e-7` — a real, expected GPU-vs-host effect, not
a bug (see `THEORY.md` "Numerical considerations" and `docs/PATTERNS.md` §4). The
`[verify]` line on stderr prints the actual `max_abs_err` so you can watch it stay
comfortably under tolerance.
