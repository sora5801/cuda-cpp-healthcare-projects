# Demo — 6.13 Gene Regulatory Network Inference

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/expression_sample.txt` input.
3. **Verify** the GPU result against the CPU reference (`reference_cpu.cpp`): the
   mutual-information matrices must agree to ~1e-9 **and** the set of pruned edges
   must be bit-identical. Prints a clear `PASS`/`FAIL`.
4. **Time** the MI and DPI kernels (CUDA events) and the CPU baseline — a
   *teaching artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## Expected result

```
6.13 -- Gene Regulatory Network Inference
ARACNE mutual-information network: 10 genes x 200 samples, 8 bins
inferred 4 direct edge(s) after MI>0.20 + DPI pruning:
  TF     -- A        I = 0.9599 nats
  TF     -- C        I = 0.9393 nats
  D      -- E        I = 0.8602 nats
  A      -- B        I = 0.7968 nats
RESULT: PASS (GPU MI matches CPU within tol=1.0e-09; edge sets identical)
```

## How to read it

The synthetic data has a **known** structure: `TF→A→B`, `TF→C`, `D→E`, and four
noise genes `F,G,H,I` (see [`../data/README.md`](../data/README.md)). Before
pruning, raw mutual information reports **seven** above-threshold edges — including
three *indirect* ones (`TF–B`, `A–C`, `B–C`) that arise only because the genes
share a common driver. The **Data Processing Inequality** step then removes each
edge that is the strictly-weakest side of some triangle, leaving exactly the
**four true direct edges** shown above. Watching those three spurious edges
disappear is the core lesson of this demo.

The `[verify]` line on stderr shows `MI max_abs_err ≈ 2.2e-16` — the GPU and CPU
build identical integer histograms and evaluate the same log sum (via the shared
`grn.h` core), so they agree to machine precision.
