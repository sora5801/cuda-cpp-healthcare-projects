# Demo — 3.29 Motif Finding in Genomic Sequences

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/sequences_sample.fasta` — 12 short
   synthetic DNA sequences, each containing one (mutated) copy of a planted
   transcription-factor motif at an unknown offset.
3. **Discover** the motif from scratch with MEME-style Expectation-Maximisation
   (no knowledge of the planted answer), reporting:
   - the **recovered consensus** motif,
   - its **information content** in bits (a standard motif-quality score), and
   - the **predicted binding site** offset in each sequence.
4. **Verify** that the GPU E-step (the parallelised window-scoring kernel) matches
   the CPU reference **exactly** (tolerance `0`), and print `PASS`/`FAIL`.
5. **Time** the GPU kernel (CUDA events) and the CPU EM — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## Expected result

```
3.29 -- Motif Finding in Genomic Sequences
MEME OOPS motif discovery: 12 sequences, width W=8, 636 windows scored
EM converged in 15 iterations
recovered motif (consensus): CTTGACGT
information content: 8.2717 bits  (max 16.0)
predicted binding site per sequence (0-based offset):
  seq[0]  site offset = 50
  ...
RESULT: PASS (GPU E-step matches CPU exactly, tol=0e+00)
```

## How to read the result

The planted motif is **`TGACGTCA`** (a CRE/AP-1-like element). EM recovers
**`CTTGACGT`** — the same biological signal captured one register to the left:
its core `TGACGT` is exactly the first six bases of the planted consensus. The
reported site offsets are correspondingly ~2 bp earlier than the planted offsets
recorded in each FASTA header. This **phase shift** is a real, well-known
behaviour of EM motif finders (the likelihood surface has near-equal optima at
neighbouring registers) — an honest teaching moment, not a bug. The
~8.3-bit information content confirms a sharp, non-random motif was found.

> The sample is **synthetic** (see `data/README.md`). Scores and sites are
> meaningful only for this toy input — no biological conclusion may be drawn.
