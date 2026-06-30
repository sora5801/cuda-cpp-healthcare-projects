# Demo — 3.6 k-mer Counting & Minimiser Sketching

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/kmer_sample.txt` input (two DNA read
   sets A and B).
3. **Compute, on both CPU and GPU**, three things and verify they agree exactly:
   - the **k-mer count histogram** of set A (canonical k-mers),
   - **minimiser MinHash sketches** of A and B,
   - the **Jaccard similarity** estimate between A and B.
4. **Print** the distinct-k-mer count, the top-8 k-mers by count (the planted
   motif tops the list), the sketch sizes, and the Jaccard estimate.
5. **Time** the kernels (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries timing and per-section verify flags (which vary run to run),
  so it is shown but never diffed.

## Expected result

```
3.6 -- k-mer Counting & Minimiser Sketching
params: k=11  w=5  s=16
set A: 12 reads, 480 bases
set B: 12 reads, 480 bases
distinct canonical k-mers in A: 203
top 8 k-mers by count:
  ACGTACGTACG  count=7
  AACGGATCGAG  count=4
  ACAGGGAATCA  count=4
  ACCAATCTACC  count=4
  AGATTGGTATG  count=4
  AGGGAATCACC  count=4
  ATACCAATCTA  count=4
  ATGACAGGGAA  count=4
sketch sizes: |A|=16  |B|=16  (bottom-16 MinHash)
Jaccard(A,B) estimate = 0.1250
RESULT: PASS (GPU hist+sketch+Jaccard match CPU exactly)
```

## How to read it

- **`ACGTACGTACG count=7`** — the planted motif (see `data/README.md`) is the
  clear winner, demonstrating that counting recovers a known signal. The other
  top k-mers (count = 4) come from overlapping regions of the planted reads.
- **`distinct canonical k-mers in A: 203`** — the number of *distinct* table
  entries the device hash table ended up holding (compacted, sorted by key).
- **`Jaccard(A,B) = 0.1250`** — the MinHash estimate of set similarity, driven by
  the ~50% genomic overlap between A's and B's sampling windows.
- **`RESULT: PASS`** — the GPU hash-table histogram, the GPU minimiser sketches,
  and the GPU Jaccard all equal the CPU reference **exactly** (integer counts and
  shared `kmer.h` math make the comparison exact, tolerance 0).

The stderr line shows the CPU vs GPU kernel milliseconds. On this tiny sample the
GPU is *launch-bound* (a few hundred k-mers) — the point here is **correctness and
the parallel pattern**, not speed. The GPU's advantage appears at the 10⁸–10⁹
k-mer scale of real read sets.
