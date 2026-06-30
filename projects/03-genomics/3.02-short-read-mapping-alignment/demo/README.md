# Demo — 3.2 Short-Read Mapping / Alignment

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/reads_sample.txt` (a tiny **synthetic**
   reference + 11 short reads).
3. **Map** every read with the GPU seed-and-extend kernel and, independently, with
   the CPU reference (`reference_cpu.cpp`), then **verify** they agree on every
   read's `(position, score, mismatches)` — exact integer equality — and print a
   clear `PASS`/`FAIL`.
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing (which varies run to run), so it is shown but
  never diffed.

## How to read the result

Each mapped read is reported as `pos P  score S  40M (k mismatches)`:

- **`pos P`** — the reference offset where the read's first base aligns. The
  synthetic reads were sampled at known, evenly spaced positions, so you can see
  the aligner recover `0, 22, 44, 66, …`.
- **`score S`** — the ungapped alignment score `= 40·MATCH + k·MISMATCH`. With
  `MATCH=+1, MISMATCH=−1` and length 40, a `k`-mismatch read scores `40 − 2k`
  (so 0/1/2/3 mismatches → 40/38/36/34). This is the embedded known answer.
- **`40M`** — a CIGAR-style "40 aligned columns, no gaps", annotated with the
  mismatch count (this teaching version does ungapped extension).
- **read 10** is deliberate random noise: its leading 12-mer is absent from the
  reference, so it is correctly reported `UNMAPPED`.

## Expected result

```
3.2 -- Short-Read Mapping / Alignment
seed-and-extend: 11 reads (L=40) vs reference (L_ref=240), seed k=12, match=+1 mismatch=-1
index: 229 reference 12-mers (sorted)
per-read mapping (read -> ref pos, score, edits):
  read  0 -> pos    0  score  40  40M (0 mismatches)
  read  1 -> pos   22  score  38  40M (1 mismatch)
  read  2 -> pos   44  score  36  40M (2 mismatches)
  read  3 -> pos   66  score  34  40M (3 mismatches)
  read  4 -> pos   88  score  40  40M (0 mismatches)
  read  5 -> pos  111  score  38  40M (1 mismatch)
  read  6 -> pos  133  score  36  40M (2 mismatches)
  read  7 -> pos  155  score  34  40M (3 mismatches)
  read  8 -> pos  177  score  40  40M (0 mismatches)
  read  9 -> pos  200  score  38  40M (1 mismatch)
  read 10 -> UNMAPPED (no seed hit)
summary: 10/11 reads mapped
RESULT: PASS (GPU matches CPU exactly on every read)
```

> The data is **synthetic** and carries no biological meaning — it exists so the
> demo runs offline with a verifiable, interpretable answer. See
> [`../data/README.md`](../data/README.md).
