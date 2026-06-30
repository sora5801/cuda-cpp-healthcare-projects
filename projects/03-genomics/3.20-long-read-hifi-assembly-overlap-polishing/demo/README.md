# Demo — 3.20 Long-Read HiFi Assembly Overlap & Polishing

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/reads_sample.txt` (12 synthetic HiFi
   reads reduced to their minimiser sketches).
3. **Verify** the GPU all-vs-all overlap scores against the CPU reference
   (`reference_cpu.cpp`) — the scores and anchor counts are integers computed by
   the *same* link function on both sides, so agreement must be **exact**
   (66/66 read pairs identical).
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the data summary (which vary run to run), so
  it is shown but never diffed.

## What you are looking at

The reads were sampled end-to-end along a synthetic genome with a 150 bp overlap
between each neighbour (and every other read reverse-complemented, to exercise
strand-symmetric seeding). So the *true* overlaps are exactly the **consecutive
read pairs** `(i, i+1)`. The program recovers precisely that:

- `candidate overlaps (chain score >= 3): 11` — the 11 neighbour pairs, and only
  those, clear the bar (all other pairs share essentially no minimisers).
- The **top-5 overlaps** are all consecutive read indices, with the chain score
  tracking how many minimisers each neighbour pair shares.

## Expected result

```
3.20 -- Long-Read HiFi Assembly Overlap & Polishing
all-vs-all overlap: 12 reads -> 66 ordered pairs scored
candidate overlaps (chain score >= 3): 11
top-5 overlaps (by chain score):
  #1  read 3 <-> read 4   score=47  anchors=47
  #2  read 4 <-> read 5   score=46  anchors=46
  #3  read 7 <-> read 8   score=45  anchors=45
  #4  read 1 <-> read 2   score=44  anchors=44
  #5  read 8 <-> read 9   score=43  anchors=43
RESULT: PASS (GPU matches CPU exactly: 66/66 pairs identical)
```

The exact scores depend on the committed sample (RNG seed 20 in
`scripts/make_synthetic.py`); regenerating the sample with different parameters
changes the numbers, so re-capture `expected_output.txt` if you do.
