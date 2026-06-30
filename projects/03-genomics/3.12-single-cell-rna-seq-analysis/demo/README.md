# Demo — 3.12 Single-Cell RNA-seq Analysis

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/scrna_sample.txt` (30 synthetic cells
   × 18 genes, 3 cell types).
3. **Verify** the GPU result against the CPU reference (`reference_cpu.cpp`): the
   KNN neighbour indices must match **exactly** (tolerance 0), and the normalized
   values + reported distances must agree within `1e-5`. Prints `PASS`/`FAIL`.
4. **Time** the two GPU kernels (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately (docs/PATTERNS.md §3):

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## What to look at in the output

- The **KNN graph**: each `cell(type) -> neighbour indices`. Notice every cell's
  neighbours share its type tag (`t0/t1/t2`) — the normalized space cleanly
  separates the three synthetic cell types.
- **`KNN label purity = 100.00%`**: of all `N·k = 150` neighbour edges, every one
  joins two same-type cells. This is the science check — the pipeline recovered
  the embedded structure despite each cell having a different sequencing depth.
- **`RESULT: PASS`**: the GPU's neighbour graph is identical to the CPU's.

## Expected result

```
3.12 -- Single-Cell RNA-seq Analysis
scRNA-seq KNN graph: 30 cells x 18 genes, k=5, target_sum=10000
pipeline: library-size normalize (counts-per-target) + log1p, then exact brute-force KNN
cell(type)  ->  k nearest neighbours [d1]
   0(t0) -> 27 18  6 12  9  [d1=7.1387]
   ...
  29(t2) -> 20 23 11  2  5  [d1=7.0000]
normalized[cell0, gene0..2] = 7.2171 7.4332 7.0688
KNN label purity = 100.00% (150/150 edges connect same-type cells)
RESULT: PASS (GPU neighbour indices match CPU exactly)
```

(The full 30-row graph is in [`expected_output.txt`](expected_output.txt).)

The `[timing]` lines on **stderr** vary per machine; on this development GPU the
kernels run in well under a millisecond. For 30 cells the GPU is *slower* than the
CPU (launch overhead dominates such a tiny O(N²)); the GPU's edge only appears at
the 10⁵–10⁷-cell scale of real data — see THEORY.md "The honest-timing rule".
