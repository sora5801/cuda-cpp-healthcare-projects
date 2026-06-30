# Demo — 2.25 Coevolutionary Contact Prediction & MSA Transformer

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/coevolution_msa.fasta` (a synthetic
   alignment with four *planted* coevolving column pairs — see `data/README.md`).
3. **Verify** the GPU raw Mutual-Information matrix against the CPU reference
   (`reference_cpu.cpp`) and print a clear `PASS`/`FAIL`.
4. **Predict contacts**: apply the Average Product Correction (APC) and rank the
   top column pairs — the four planted contacts should appear as #1–#4.
5. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## Expected result

```
2.25 -- Coevolutionary Contact Prediction & MSA Transformer
MSA: 400 sequences x 24 columns (alphabet Q=21: 20 aa + gap)
Method: pairwise Mutual Information + Average Product Correction (APC)
Top 8 predicted contacts (1-based columns, by APC score):
  #1  ( 4,  5)  APC=1.391032  rawMI=1.535621
  #2  ( 9, 16)  APC=1.377826  rawMI=1.532141
  #3  ( 3, 22)  APC=1.370339  rawMI=1.514990
  #4  ( 6, 19)  APC=1.327577  rawMI=1.455841
  #5  (17, 23)  APC=0.132486  rawMI=0.574875
  #6  ( 2, 24)  APC=0.128479  rawMI=0.571221
  #7  ( 2, 11)  APC=0.124321  rawMI=0.569834
  #8  ( 2, 12)  APC=0.123910  rawMI=0.573596
RESULT: PASS (GPU MI matrix matches CPU within tol=1.0e-09 nats)
```

### How to read it

- **Ranks #1–#4 are the four planted contacts** `(4,5) (9,16) (3,22) (6,19)`
  (data/README.md). Their APC scores (~1.3–1.4) sit an order of magnitude above
  the best decoy (~0.13 at rank #5), so the prediction is unambiguous — exactly
  what a useful contact predictor should produce.
- **`APC` vs `rawMI`**: `rawMI` is the uncorrected Mutual Information; `APC` is
  after subtracting the per-column background (Average Product Correction). Notice
  the decoys' raw MI (~0.57) is *not* tiny — high-entropy columns accumulate
  spurious MI with everyone — but APC shrinks it to ~0.13, which is what separates
  the real signal from the noise.
- **`RESULT: PASS`** means the GPU's MI matrix equals the CPU reference's to within
  `1e-9` nats (it actually agrees to ~`4e-16`, machine precision; see the stderr
  `[verify]` line). The two compute identical integer counts and call the same
  `cv_mi_from_counts`, so they cannot meaningfully diverge.

The stderr timing shows the CPU is *faster* here: with L=24 there are only 276
column pairs, far too few to amortize the kernel launch + PCIe copies. That is
honest and expected (PATTERNS.md §7); the GPU's advantage grows with L (real MSAs
have L in the hundreds to >1000, i.e. 10⁴–10⁶ independent pairs).
