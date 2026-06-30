# Demo — 2.13 MSA Generation Acceleration

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/profile_db_sample.txt` — one query
   **profile HMM** (length L = 10) searched against **24 synthetic database
   sequences**.
3. **Verify** the GPU Viterbi scores against the CPU reference
   (`reference_cpu.cpp`) and print a clear `PASS`/`FAIL`. Because every score is a
   scaled **integer** log-odds and both sides run the *same* integer recurrence,
   the agreement is **exact** (`max |diff| = 0`), not approximate.
4. **Report** the **top-5 hits** by Viterbi log-odds score, and **time** the
   kernel (CUDA events) against the CPU baseline — a *teaching artifact*, never a
   benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the score difference (which vary run to run),
  so it is shown but never diffed.

## Expected result

See [`expected_output.txt`](expected_output.txt) for the exact stdout the demo
asserts:

```
2.13 -- MSA Generation Acceleration
Profile-HMM Viterbi search: 1 query profile (L=10) vs 24 database sequences
top-5 hits (by Viterbi log-odds score):
  #1  seq[0]  score = 20000  (log-odds = 20.000)
  #2  seq[1]  score = 17500  (log-odds = 17.500)
  #3  seq[2]  score = 15500  (log-odds = 15.500)
  #4  seq[10]  score = 6000  (log-odds = 6.000)
  #5  seq[17]  score = 5000  (log-odds = 5.000)
RESULT: PASS (GPU matches CPU exactly; max |diff| = 0)
```

## Reading the result (the embedded "known answer")

The synthetic sample is **engineered** so the answer is checkable: sequences
`0`, `1`, `2` each contain the query's motif (`WYGGFPKDEC`), so they should be the
**top three hits** — and they are. `seq[0]` carries the clean motif (highest
score, `20.000`); `seq[1]` has one point mutation (`17.500`); `seq[2]` has a
2-residue insertion that the HMM's insert states absorb at a transition cost
(`15.500`). The remaining hits (`seq[10]`, `seq[17]`) are pure-noise sequences
that happen to align a few residues by chance — exactly the low-scoring
background a real search filters out.

> The scores reflect the **synthetic** sample (a made-up motif and fabricated
> log-odds); they carry **no biological meaning**. This is study material, not a
> tool for real MSA construction.
