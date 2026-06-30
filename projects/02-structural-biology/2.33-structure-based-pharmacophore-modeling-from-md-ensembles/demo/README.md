# Demo — 2.33 Structure-Based Pharmacophore Modeling from MD Ensembles

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/pharmacophore_sample.txt` input.
3. **Verify** the GPU per-molecule scores against the CPU reference
   (`reference_cpu.cpp`) and print a clear `PASS`/`FAIL`.
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately (PATTERNS.md §3):

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## What you are looking at

The demo screens **one query pharmacophore** (5 typed 3-D feature points — the
consensus an MD ensemble would yield) against **512 library molecules**, scoring
each with a ROCS-style Gaussian "color" Tanimoto. The synthetic sample plants a
near-perfect match at **molecule 7** (a sub-angstrom-jittered copy of the query),
so the headline check is that **mol[7] ranks #1**, well above the random decoys.

The `score = 0.59` for the planted hit (rather than ~1.0) is expected and honest:
the planted molecule carries one or two extra decoy features that inflate its
self-overlap, which the Tanimoto denominator penalizes — exactly how a real ROCS
score behaves for an imperfect but clearly-best match.

## Expected result

```
2.33 -- Structure-Based Pharmacophore Modeling from MD Ensembles
ensemble pharmacophore: 5 features
  feature[0] aromatic   at (10.39, 10.28, 9.38) weight 0.78
  feature[1] pos-charge at (2.28, 9.65, 5.71) weight 0.85
  feature[2] acceptor   at (1.13, 3.64, 1.09) weight 0.92
  feature[3] neg-charge at (7.61, 7.14, 4.75) weight 0.78
  feature[4] neg-charge at (7.39, 1.89, 0.18) weight 0.81
screen: 1 pharmacophore vs 512 library molecules
top-5 hits (ROCS-style color Tanimoto):
  #1  mol[7]  score = 0.593165
  #2  mol[147]  score = 0.108589
  #3  mol[280]  score = 0.101253
  #4  mol[444]  score = 0.096770
  #5  mol[102]  score = 0.072042
planted target mol[7] score = 0.593165, rank = 1 of 512
RESULT: PASS (GPU matches CPU within tol=1.0e-05)
```

stderr will additionally show the data source, CPU/GPU timing, and the verify line
`max_abs_err = 0.000e+00` — the GPU reproduces the CPU score exactly here because
both call the identical `score_molecule()` (see `src/pharmacophore.h`).
