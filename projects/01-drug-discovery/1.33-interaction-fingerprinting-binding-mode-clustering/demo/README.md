# Demo — 1.33 Interaction Fingerprinting & Binding-Mode Clustering

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/ifp_sample.txt` (120 poses in a
   24-residue model pocket, drawn from 4 planted binding modes).
3. **STAGE A** — build the interaction fingerprints on CPU and GPU and confirm
   they are **bit-identical**.
4. **STAGE B** — cluster the IFPs into binding modes on CPU and GPU and confirm
   the labels + consensus centroids match **exactly**.
5. **Report** each mode's consensus contacts, the clustering cost, and the
   recovery purity vs. the planted modes; print a clear `PASS`/`FAIL`.
6. **Time** both stages (CUDA events) — a *teaching artifact*, not a benchmark.

The program splits its output deliberately (PATTERNS.md §3):

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt). Determinism is guaranteed by
  using only integer math: geometry→bits is exact, and the consensus centroid is
  an integer **majority vote** (no float atomics to reorder).
- **stderr** carries the timing (which varies run to run), so it is shown but
  never diffed.

## Expected result

```
1.33 -- Interaction Fingerprinting & Binding-Mode Clustering
pocket: 24 residues x 4 interaction types = 96 IFP bits
stage A: built 120 interaction fingerprints (CPU==GPU: yes)
stage B: 120 poses -> 4 binding-mode clusters, 12 iterations
  cluster 0 (n=  30): consensus contacts = R0:hydrophobic R0:hbond R1:hydrophobic R4:hydrophobic R4:hbond R5:hydrophobic
  cluster 1 (n=  30): consensus contacts = R2:hydrophobic R3:hydrophobic R6:hydrophobic R7:hydrophobic R7:aromatic
  cluster 2 (n=  30): consensus contacts = R16:hydrophobic R16:hbond R17:hydrophobic R17:ionic R20:hydrophobic R20:hbond R20:ionic R21:hydrophobic
  cluster 3 (n=  30): consensus contacts = R18:hydrophobic R19:hydrophobic R22:hydrophobic R23:hydrophobic R23:ionic
cost = 2.7484
mode recovery (purity vs planted modes) = 100.00%
RESULT: PASS (GPU IFPs + labels + centroids match CPU)
```

## How to read it

- **Four clusters of 30** — the demo recovered all four planted modes exactly
  (purity 100%). Each cluster's "consensus contacts" lists the residue:type bits
  set in a majority of its members — a human-readable summary of *that binding
  mode's interaction pattern*. The four lists touch four disjoint residue
  neighborhoods (top-left, top-right, bottom-left, bottom-right of the pocket),
  exactly the four corners the synthetic modes were planted in.
- **`cost`** is the k-means objective in Tanimoto space (sum of each pose's
  distance to its mode's consensus); lower means tighter modes.
- **`CPU==GPU: yes` / `RESULT: PASS`** is the correctness gate: the GPU IFPs,
  cluster labels, and consensus centroids all match the CPU reference bit-for-bit.
