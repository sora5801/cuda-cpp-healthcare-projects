# Demo — 7.2 Drug-Target Interaction Prediction (GNN)

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/dti_sample.txt` input.
3. **Verify** the GPU result against the CPU reference (`reference_cpu.cpp`) and
   print a clear `PASS`/`FAIL`. Both run the *same* fixed-weight message-passing
   GNN forward pass, so they must agree to within a tiny floating-point tolerance.
4. **Time** the kernels (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic (fixed seeded weights + fixed-order
  reductions) and is diffed against [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric verification error (which vary
  run to run), so it is shown but never diffed.

## Expected result

```
7.2 -- Drug-Target Interaction Prediction (GNN)
[reduced-scope teaching model: FIXED (untrained) message-passing GNN]
batch: 6 drugs x 4 proteins, 24 atoms total, F=8, T=2 rounds
DTI score matrix (rows = drugs, cols = proteins), probabilities:
  drug 0: 0.5282 0.5182 0.5285 0.5579
  drug 1: 0.5448 0.5283 0.5444 0.5898
  drug 2: 0.5645 0.5383 0.5611 0.6260
  drug 3: 0.5355 0.5203 0.5330 0.5705
  drug 4: 0.5550 0.5308 0.5501 0.6068
  drug 5: 0.5972 0.5448 0.5780 0.6820
top interaction: drug 5 <-> protein 3  (score 0.6820)
implanted ground truth: drug 5 <-> protein 3  -> RECOVERED
RESULT: PASS (GPU embeddings+scores match CPU)
```

## How to read it

- Each row is one drug; each column one protein target. Cells are the model's
  interaction **probability** (sigmoid of the drug–protein embedding dot product).
- `top interaction` is the argmax over the whole matrix — the model's single best
  DTI prediction. It matches the `implanted ground truth`, so the pipeline
  (message passing → pooling → protein encoding → scoring) recovered the pair the
  synthetic data was built around: `RECOVERED`.
- The scores are illustrative of the *machinery* (the weights are **untrained**);
  they carry no clinical meaning.
