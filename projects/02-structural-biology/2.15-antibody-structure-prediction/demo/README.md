# Demo — 2.15 Antibody Structure Prediction (reduced-scope: CDR screening)

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/antibodies_sample.txt` (1 query +
   24 synthetic library antibodies).
3. **Screen** the library against the query by CDR-weighted BLOSUM62 similarity
   on the **GPU** (`src/kernels.cu`), and independently on a **CPU reference**
   (`src/reference_cpu.cpp`).
4. **Verify** the GPU scores match the CPU scores **exactly** (integer math,
   tolerance 0) and print a clear `PASS`/`FAIL`.
5. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## What to look for

The screen should recover the **planted answer**: `mAb_07` (a near-copy of the
query) ranks **#1**, and `mAb_18` (which shares the query's exact CDR-H3) ranks
**#2** — far above every other antibody. The per-hit breakdown shows how much of
each score comes from the ×3-weighted **CDR-H3** loop, making its dominance
visible.

## Expected result

```
Antibody Structure Prediction (reduced: CDR screening)
catalog ID 2.15 -- reduced-scope teaching version
query antibody: query_Ab
screened 24 library antibodies by CDR-weighted BLOSUM62 similarity
(CDR-H3 weighted x3; higher score = more similar CDRs)
top-5 hits:
  #1  mAb_07      score =   516  (CDR-H3 contributes  303)
  #2  mAb_18      score =   373  (CDR-H3 contributes  354)
  #3  mAb_04      score =    25  (CDR-H3 contributes   -9)
  #4  mAb_17      score =    23  (CDR-H3 contributes  -15)
  #5  mAb_02      score =     1  (CDR-H3 contributes   -9)
RESULT: PASS (GPU matches CPU exactly, integer scores)
```

> **Scope note:** this is the reduced-scope teaching version (CLAUDE.md §13). It
> performs antibody library **screening by CDR similarity**, not 3-D structure
> prediction. See the project `README.md` and `THEORY.md` for what the full
> IgFold/ABodyBuilder3 pipeline does and how this relates to it.
