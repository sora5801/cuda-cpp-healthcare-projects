# Demo — 3.9 Phylogenetic Likelihood / Tree Inference

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/phylo_sample.txt` — a synthetic
   8-taxon DNA alignment evolved down a **known** tree.
3. **Score** three candidate trees by their total log-likelihood (the sum over
   600 sites of Felsenstein's pruning recursion) on **both** the GPU
   (`src/kernels.cu`) and the CPU reference (`src/reference_cpu.cpp`).
4. **Verify** that the GPU's per-tree log-likelihoods match the CPU's **exactly**
   (both reduce the same fixed-point integers), and print `PASS`/`FAIL`.
5. **Time** the kernels (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## What to look for

The **true** generating tree (`..._true`) gets the **highest** log-likelihood
(closest to zero, since lnL is negative) and is announced as the maximum-likelihood
tree; the two wrong rearrangements (`wrong_NNI1`, `wrong_NNI2`) score noticeably
lower. That is the whole idea of ML phylogenetics in miniature: the data prefer the
topology they were generated under. The `RESULT: PASS` line confirms the GPU and
CPU agree to the last bit.

## Expected result (stdout)

```
3.9 -- Phylogenetic Likelihood / Tree Inference
alignment: 8 taxa x 600 sites   model: K2P (kappa = 2.00)
candidate trees scored: 3
  tree[0] ((t0,t1),(t2,t3)),((t4,t5),(t6,t7))_true lnL = -3341.650694
  tree[1] wrong_NNI1             lnL = -4251.060632
  tree[2] wrong_NNI2             lnL = -4434.280987
MAXIMUM-LIKELIHOOD TREE: tree[0] ((t0,t1),(t2,t3)),((t4,t5),(t6,t7))_true  (lnL = -3341.650694)
RESULT: PASS (GPU per-tree lnL match CPU exactly)
```

The exact `lnL` values are reproducible because the sample is generated with a
fixed PRNG seed and the likelihood is summed in deterministic fixed-point. Timings
on stderr will differ on your machine.
