# Demo — 3.8 Multiple Sequence Alignment (MSA)

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed synthetic family in `data/sample/`.
3. **Verify** the GPU pairwise-score matrix against the CPU reference
   (`reference_cpu.cpp`) and print a clear `PASS`/`FAIL`.
4. **Time** the GPU kernel (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and run-varying detail, so it is shown but never
  diffed.

## What you are looking at

```
3.8 -- Multiple Sequence Alignment (MSA)
input: 6 DNA sequences, max length 27; scoring match=+2 mismatch=-1 gap=-2
pairwise NW alignments (STAGE 1) = 15  (one GPU block each)
center-star sequence (STAGE 2) = index 1 ("seq1")
multiple alignment (STAGE 3): 6 rows x 28 columns, Sum-of-Pairs score = 553

seq0       -T-GTACGTACGTT-G-CAACGT-ATCG
seq1       -TAGTACGTACGTT-G-CAACGT-ATCG  <- center
seq2       TTAGTACGTACGTT-G-C--CGT-ATCG
seq3       -TAGTACGTACGTT-G-CAACGT-ATCG
seq4       -GAGTACGTCCGTTCGCCAACGTAATCG
seq5       -TAGTACGTACGAT-GTCAACGT-ATAG
conserv.      ****** ** * * *  *** ** *

RESULT: PASS (GPU pairwise-score matrix matches CPU exactly)
```

- **STAGE 1** runs 15 independent Needleman-Wunsch alignments (one per pair of the
  6 sequences), each on its own **GPU thread block** — the lesson of this project.
- **STAGE 2** picks the center-star sequence (`seq1`, the most representative).
- **STAGE 3** folds every sequence onto the center, producing the aligned block.
  The `conserv.` line marks columns where **all six** rows carry the same
  nucleotide with `*` — those stars are the recovered conserved core, the visible
  proof the alignment is correct.
- **RESULT: PASS** means the GPU score matrix equals the CPU one **exactly**
  (integer scores, tolerance `0`).

The timing line on **stderr** (e.g. `GPU STAGE-1 kernel: 0.25 ms`) varies per run
and is shown for illustration only — on this tiny input the GPU is launch-bound
and the lesson is the *mapping*, not the speed (CLAUDE.md §12; PATTERNS.md §7).
