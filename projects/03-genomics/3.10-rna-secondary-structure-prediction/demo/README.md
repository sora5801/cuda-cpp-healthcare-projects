# Demo — 3.10 RNA Secondary-Structure Prediction

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/rna_sample.fasta` input.
3. **Fold** the RNA two ways — a serial CPU Nussinov DP (`reference_cpu.cpp`) and
   the GPU anti-diagonal wavefront (`kernels.cu`) — and **verify** that the two
   dynamic-programming matrices are **identical, cell for cell** (exact integer
   equality), printing a clear `PASS`/`FAIL`.
4. **Time** the GPU wavefront (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic (the sequence, the predicted
  dot-bracket structure, the max base-pair count) and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the mismatch counters (which vary run to
  run), so it is shown but never diffed.

## Reading the result

The committed sample is an 18-nt synthetic **hairpin**, `GGGCGCAAAAGCGCCCAU`.
Nussinov maximises the number of (non-crossing) base pairs, and the optimal
folding is a 6-bp stem closing an `AAAA` loop:

```
sequence : GGGCGCAAAAGCGCCCAU
structure: ((((((....))))))..
max base pairs = 6
```

In the dot-bracket notation, each `(` pairs with the matching `)` (counting
nesting), and `.` is an unpaired base. Six `(`/`)` pairs = six base pairs, exactly
the designed answer (see `data/README.md`). `RESULT: PASS` means the GPU matrix
matched the CPU matrix everywhere — the correctness guarantee.

## Expected stdout

```
3.10 -- RNA Secondary-Structure Prediction (Nussinov)
RNA length n = 18  (alphabet ACGU, min hairpin loop = 3)
sequence : GGGCGCAAAAGCGCCCAU
structure: ((((((....))))))..
max base pairs = 6
RESULT: PASS (GPU matrix matches CPU exactly)
```
