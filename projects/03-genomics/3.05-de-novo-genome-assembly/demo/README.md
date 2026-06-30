# Demo — 3.5 De Novo Genome Assembly

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/reads_sample.fasta` (6 synthetic reads).
3. **Sketch** each read into its minimizer set, then compute the **all-vs-all
   shared-minimizer count** for every read pair on **both** the CPU reference and
   the GPU kernel.
4. **Verify** that the GPU's per-pair scores match the CPU's **exactly** (integer
   counts ⇒ tolerance `0`) and print a clear `PASS`/`FAIL`.
5. **Report** the resulting **overlap graph**: the edges (pairs sharing ≥ 3
   minimizers) and its connected-component structure (≈ contigs).
6. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately (PATTERNS.md §3):

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the mismatch count (which vary run to run), so
  it is shown but never diffed.

## Expected result

```
3.5 -- De Novo Genome Assembly
all-vs-all read overlap via minimizers (k=15, w=5)
reads = 6   pairs = 15   total minimizers = 76
overlap edges (shared minimizers >= 3):
  read 0 -- read 1   shared = 10
  read 0 -- read 2   shared = 5
  read 0 -- read 3   shared = 3
  read 1 -- read 2   shared = 9
  read 1 -- read 3   shared = 7
  read 1 -- read 4   shared = 3
  read 2 -- read 3   shared = 9
  read 2 -- read 4   shared = 5
  read 3 -- read 4   shared = 8
  read 3 -- read 5   shared = 5
  read 4 -- read 5   shared = 10
graph: 11 edge(s), 1 component(s), largest component = 6 read(s)
RESULT: PASS (GPU per-pair scores match CPU exactly, tol=0)
```

## How to read it

- The reads were tiled from one pseudo-genome (positions 0,12,…,60), so they
  form **one chain**: every read overlaps its immediate neighbours strongly
  (e.g. `read 0 -- read 1  shared = 10`) and overlap fades with distance
  (`read 0 -- read 3  shared = 3`, and `0 -- 5` is below threshold, so absent).
- **`1 component(s), largest component = 6`** is the punchline: all six reads
  belong to one cluster ⇒ they assemble into a **single contig**. That is the
  layout step of de-novo assembly, recovered from the GPU's overlap scores.
- `RESULT: PASS … tol=0` means the GPU and CPU agreed on **every** pair's count —
  exact because the shared math is integer (`src/assembly.h::count_shared_sorted`).
