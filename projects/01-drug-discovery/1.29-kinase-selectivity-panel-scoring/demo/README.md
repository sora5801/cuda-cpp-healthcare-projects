# Demo — 1.29 Kinase Selectivity Panel Scoring

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/kinase_panel_sample.txt` — one compound
   profiled against a 16-kinase panel.
3. **Verify** the GPU result against the CPU reference (`reference_cpu.cpp`)
   **exactly** (integer physics, tolerance 0) and print a clear `PASS`/`FAIL`.
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the exact-match mismatch count (which vary run
  to run, or carry run-specific paths), so it is shown but never diffed.

## What the learner is seeing

- **`S-score(pK>=6.000) = 1/16 = 0.062`** — only 1 of the 16 kinases is bound above
  the `pK ≥ 6` (≈ 1 µM) threshold, so this compound is **selective**. A small
  S-score is the goal in kinase drug design.
- **`top-5 most potently bound kinases`** — `ABL1` is the unique top hit (the
  intended target, by construction of the synthetic data), followed by a cluster of
  near-threshold off-targets (`SRC`, `PDGFRA`, `KIT`, `LCK`). That cluster is the
  whole point: kinases share an ATP pocket, so the runner-ups sit *just* below the
  hit line — exactly why selectivity is hard.
- **`RESULT: PASS (GPU matches CPU exactly ...)`** — the GPU kernel and the serial
  CPU reference produced identical per-kinase pK values, identical hit flags, and the
  same S-count.

## Expected result

```
1.29 -- Kinase Selectivity Panel Scoring
panel: 1 compound vs 16 kinases (8-feature interaction fingerprint)
S-score(pK>=6.000) = 1/16 = 0.062  (lower = more selective)
top-5 most potently bound kinases:
  #1  ABL1        pK = 6.050  [HIT]
  #2  SRC         pK = 5.950
  #3  PDGFRA      pK = 5.750
  #4  KIT         pK = 5.650
  #5  LCK         pK = 5.500
RESULT: PASS (GPU matches CPU exactly: per-kinase pK, hit flags, S-count)
```

The data is **synthetic** (see [`../data/README.md`](../data/README.md)); the result
demonstrates the *method*, not real pharmacology.
