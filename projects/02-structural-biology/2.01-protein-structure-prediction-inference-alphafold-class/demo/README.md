# Demo — 2.1 Protein Structure Prediction Inference (AlphaFold-class)

> **Reduced-scope teaching version:** the demo runs **one scaled dot-product
> self-attention head** over a tiny synthetic protein, the core building block of
> the Evoformer/ESMFold transformer stacks (see [THEORY.md](../THEORY.md)).

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/attention_sample.txt` input.
3. **Verify** the GPU attention result against the CPU reference
   (`reference_cpu.cpp`) and print a clear `PASS`/`FAIL`.
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## Expected result

```
2.1 -- Protein Structure Prediction Inference (AlphaFold-class)
[reduced-scope: one scaled dot-product self-attention head]
L = 6 residues, d = 32 feature channels
per-residue attention (GPU result):
  residue 0 -> attends most to residue 0 (w=0.487150)  |out|=2.552645
  residue 1 -> attends most to residue 1 (w=0.502807)  |out|=2.895202
  residue 2 -> attends most to residue 2 (w=0.495691)  |out|=3.312552
  residue 3 -> attends most to residue 3 (w=0.490794)  |out|=3.695886
  residue 4 -> attends most to residue 4 (w=0.504296)  |out|=4.108133
  residue 5 -> attends most to residue 5 (w=0.499159)  |out|=4.493561
RESULT: PASS (GPU matches CPU within tol=1.0e-05)
```

## How to read it

- Each line reports, for one query residue, **which residue it attends to most**
  and that softmax weight `w`, plus the **L2 norm** of its context-mixed output
  vector. Because the synthetic data gives every residue a unique "identity"
  channel, each residue attends most to **itself** — the embedded known answer
  (see [`data/README.md`](../data/README.md)). The dominant weight is ~0.5 (not
  ~1.0) because the small ramp features make the other residues weakly similar
  too — exactly what softmax should do.
- `RESULT: PASS` means the GPU kernel's full output matrix matched the CPU
  reference within `1e-5` (the stderr line shows the actual `max_abs_err`,
  ~`4.8e-7` here).

> The numbers above were **captured from a real run** on the build machine
> (RTX 2080, sm_75). They are deterministic, so your `stdout` should match
> byte-for-byte; only the stderr timing differs.
