# Demo — 3.7 BLAST-Style Homology Search

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/proteins_sample.fasta` (a synthetic
   query + 9-sequence protein database).
3. **Search** the query against every DB sequence with the BLAST-style
   seed-filter-extend pipeline (k-mer prefilter + gapless X-drop extension,
   BLOSUM62), on the **GPU** (one thread per DB sequence) and on the **CPU**.
4. **Verify** the GPU result against the CPU reference (`reference_cpu.cpp`):
   because every score is an integer computed by the *same* shared code
   (`blast_core.h`), the two agree **exactly** (`max integer score diff = 0`).
5. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the verify diff (which vary run to run, or
  contain the absolute input path), so it is shown but never diffed.

## Expected result

```
3.7 -- BLAST-Style Homology Search
query: QUERY  (len=120)
seed-extend search: 1 query vs 9 DB sequences (k=4, X-drop=12, BLOSUM62)
top-5 homology hits (by best ungapped HSP score):
  #1  db[0]  hit_close   HSP_score = 723
  #2  db[1]  hit_medium  HSP_score = 538
  #3  db[2]  hit_domain  HSP_score = 257
  #4  db[3]  decoy_1     HSP_score = 0
  #5  db[4]  decoy_2     HSP_score = 0
RESULT: PASS (GPU matches CPU exactly, integer scores)
```

## How to read it

The synthetic database is engineered with **known** relationships (see
`data/README.md`), and the result recovers them in the correct order:

- **`hit_close`** (the query with ~8% mutated) scores highest (723) — a strong
  near-full-length ungapped HSP.
- **`hit_medium`** (~25% mutated) is clearly related but lower (538) — divergence
  costs score.
- **`hit_domain`** (an unrelated scaffold with the query's middle 40 residues
  spliced in) scores 257 from **one local HSP over the shared domain** even
  though its flanks are random. This is the whole point of *local* seed-extend
  search: it finds shared regions, not just whole-sequence similarity.
- **decoys** (fully random proteins) score 0 — no seed survives extension.

So a single run shows both that the GPU and CPU agree exactly *and* that the
algorithm recovers the designed biology. The HSP scores are in BLOSUM62 units
(higher = more similar); they are **not** E-values (see THEORY for what a real
tool adds on top).
