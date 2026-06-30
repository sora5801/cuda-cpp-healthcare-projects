# Demo — 3.30 Pangenome Graph Construction

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/pangenome_sample.txt` graph.
3. **Verify** the GPU layout against the CPU reference (`reference_cpu.cpp`) —
   identical node positions and identical 1-D node order — and print `PASS`/`FAIL`.
4. **Time** the GPU sweep loop (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the verify diagnostics (which vary run to
  run), so it is shown but never diffed.

## How to read the result

- **`node coordinates`** — each node's final 1-D position (bp), leftmost node
  anchored at 0. Watch where the variant nodes land: node **10** (the SNP
  alternate to node 4) sits right beside node 4, and node **11** (the insertion)
  sits between nodes 6 and 7.
- **`1-D node order`** — the permutation that sorts nodes by coordinate. This is
  the `odgi sort` deliverable: `0 1 2 3 10 4 5 6 11 7 8 9`.
- **`stress`** — the layout objective before and after. It falls ~100× (from
  ~2.23M to ~18.9k), showing the layout converged.
- **`RESULT: PASS`** — the GPU and CPU agree exactly (same positions to 1e-9 bp,
  same order).

## Expected result (stdout)

```
3.30 -- Pangenome Graph Construction
graph: 12 nodes, 4 genome paths -> 39 layout terms
layout: 100 SMACOF sweeps (hops=3, Guttman transform)
node coordinates (bp, leftmost node at 0):
  node  0 (len  300 bp): x =      0.000
  node  1 (len  150 bp): x =    299.450
  node  2 (len  200 bp): x =    450.104
  node  3 (len  250 bp): x =    652.354
  node  4 (len  180 bp): x =    919.888
  node  5 (len  220 bp): x =   1028.029
  node  6 (len  160 bp): x =   1165.684
  node  7 (len  240 bp): x =   1367.889
  node  8 (len  190 bp): x =   1603.661
  node  9 (len  300 bp): x =   1794.964
  node 10 (len  180 bp): x =    857.816
  node 11 (len  120 bp): x =   1285.264
1-D node order (left to right): 0 1 2 3 10 4 5 6 11 7 8 9
stress: initial 2234207.1515 -> final 18893.8519
RESULT: PASS (GPU layout matches CPU; same 1-D order)
```

The stderr (timing + verify) is shown by the demo but not diffed; the GPU loop
time is launch-bound on this tiny graph and will vary run to run.
