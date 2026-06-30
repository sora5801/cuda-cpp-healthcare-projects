# Demo — 3.13 Pangenome Graph Alignment

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/graph_sample.txt` (a tiny synthetic
   variation graph + one read).
3. **Verify** the GPU result against the CPU reference (`reference_cpu.cpp`):
   every cell of every per-node score block must match **exactly** (integer DP),
   printing a clear `PASS`/`FAIL`.
4. **Time** the per-node wavefront (CUDA events) and the CPU baseline — a
   *teaching artifact*, not a benchmark claim.

The program splits its output deliberately (PATTERNS.md §3):

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the cell-mismatch count (which vary run to
  run), so it is shown but never diffed.

## What to look at in the output

```
best local score = 96  ending at node a4, cell (i,j)=(54,6)
best path through graph = a0>s0ref>a1>s1alt>a2>s2ref>a3>s3alt>a4
aligned length = 54, identities = 50/54 (92.6%)
  Q: GCTAATTACAATACATAATATTCACGTCAGCACGAAACTTGTTGGACCGTTTGA
     |||||..|||||||||||.|||||||||||||||||||||||||||||||.|||
  G: GCTAAAGACAATACATAACATTCACGTCAGCACGAAACTTGTTGGACCGTGTGA
RESULT: PASS (GPU blocks match CPU exactly)
```

The headline is the **best path through graph**: the read was synthesised to
follow the `ref` allele on even SNP bubbles and the `alt` allele on odd ones
(`data/README.md`), and the aligner recovers exactly that path
(`a0>s0ref>a1>s1alt>a2>s2ref>a3>s3alt>a4`). The `.` markers are the few seeded
point mutations (4 of them → 92.6 % identity). `RESULT: PASS` confirms the GPU's
per-node wavefront fill reproduced the CPU reference cell-for-cell.

## Expected result

The committed [`expected_output.txt`](expected_output.txt) was **captured from a
real run** on an NVIDIA RTX 2080 (Turing, `sm_75`), CUDA 13.3 + VS 2026. The
stdout is deterministic, so it should match on any machine; the stderr timings
will differ and are not compared.
