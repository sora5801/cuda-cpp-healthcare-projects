# Demo — 3.23 Splice-Aware RNA Alignment

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/reads_sample.txt` (6 synthetic reads
   vs a 3-exon reference gene model).
3. **Align** every read with the **splice-aware DP** on both the CPU reference
   and the GPU, and **verify** they agree to the integer — scores, endpoints,
   *and every DP-table cell* (`RESULT: PASS`).
4. **Time** the GPU kernel (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic (the per-read CIGARs + the junction
  summary) and is diffed against [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timings and verify counters (which vary run to run), so
  it is shown but never diffed.

## How to read the output

Each read prints its best spliced-local-alignment **score**, the **endpoint**
cell `(i,j)` in the DP table, the number of **introns** it crossed, and its
**CIGAR**. The CIGAR's `N` operation is the splice-aware payoff: it means
"skip this many reference bases (an intron) for free", instead of paying a
per-base gap penalty. So:

- `20M` — 20 matched bases, no intron (an exon-internal read).
- `12M40N12M` — 12 matches, **skip a 40-base intron**, 12 more matches: a read
  spanning one exon-exon junction.
- `6M40N24M48N6M` — a read crossing **two** junctions (both introns skipped).

The closing `junction summary` line counts how many reads crossed ≥1 intron.

## Expected result

```
3.23 -- Splice-Aware RNA Alignment
reference gene model: N=172 bases, reads=6, max read len=36
scoring: match=+2 mismatch=-1 gap=-2 intron_open=-6 canonical(GT-AG)_bonus=+4
per-read spliced alignment (CIGAR uses N for intron skips):
  read  0: len= 20 score= 40 end=(i= 20,j= 25) introns=0  CIGAR=20M
  read  1: len= 22 score= 44 end=(i= 22,j=167) introns=0  CIGAR=22M
  read  2: len= 24 score= 46 end=(i= 24,j= 82) introns=1  CIGAR=12M40N12M
  read  3: len= 24 score= 46 end=(i= 24,j= 86) introns=1  CIGAR=8M40N16M
  read  4: len= 24 score= 46 end=(i= 24,j=156) introns=1  CIGAR=10M48N14M
  read  5: len= 36 score= 68 end=(i= 36,j=148) introns=2  CIGAR=6M40N24M48N6M
junction summary: 4/6 reads cross >=1 intron, 5 intron(s) detected total
reference[0:60] = AGACTTTCAAAGATATGCTGGGTAGAGGTCGTACCCACCACCCAACACAAACAAAAACCA
RESULT: PASS (GPU matches CPU exactly: scores, endpoints, all DP cells)
```

The `stderr` lines (timings, `[verify] ... mismatches=0 ...`) appear in the demo
console but are not part of the diff.
