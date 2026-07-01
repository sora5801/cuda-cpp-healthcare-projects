# Demo — 4.22 Quantitative Susceptibility Mapping (QSM)

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/field_map.txt` — a synthetic 16×16×8
   field map produced from a known susceptibility phantom.
3. **Reconstruct** the susceptibility map χ two ways on the GPU with cuFFT (TKD and
   iterative Tikhonov), and the same three ways on the CPU (direct DFT).
4. **Verify** three things and print a clear `PASS`/`FAIL`:
   - GPU TKD matches the CPU TKD (to ~`1e-16`),
   - GPU iterative matches the CPU iterative (to ~`1e-16`),
   - the iterative solve converged to the closed-form Tikhonov minimizer.
5. **Time** the cuFFT reconstructions (CUDA events) against the O(N²) CPU DFT — a
   *teaching artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric errors (which vary run to run), so
  it is shown but never diffed.

## What to look for

- **`recovered chi at sources`** vs **`ground-truth chi at sources`** — the
  reconstruction recovers the four phantom blobs, including the *sign* of the
  diamagnetic one (`−0.7`). Note the recovered values are a bit **smaller** than
  the truth: that is the real QSM regularization bias, not an error.
- **`chi RMS vs ground truth`** — a single global recovery score per method.
- **`data-consistency residual`** — re-applying the forward dipole model to the
  reconstructed χ reproduces the input field map to a small residual.
- On **stderr**, the CPU (direct DFT) takes ~500 ms while the GPU (cuFFT) takes a
  couple of ms — the O(N²) → O(N log N) win, which explodes with volume size.

## Expected result (stdout)

```
4.22 -- Quantitative Susceptibility Mapping (QSM)
volume: 16x16x8 voxels   B0 || z   dipole kernel D(k)=1/3 - kz^2/|k|^2
methods: TKD(thr=0.15)  Tikhonov(alpha=0.05, closed-form + 200-iter GD, step=0.50)
TKD  recovered chi at sources: +0.8320 +0.4987 +0.6677 -0.5849
TIK  recovered chi at sources: +0.4381 +0.2643 +0.3509 -0.3055
ground-truth chi at sources: +1.0000 +0.6000 +0.8000 -0.7000
chi RMS vs ground truth:  TKD=0.0115  Tikhonov=0.0217
data-consistency residual (forward-model refit): 0.0006
RESULT: PASS (GPU cuFFT inversion matches CPU reference; iterative solve converged)
```

(The exact bytes are in [`expected_output.txt`](expected_output.txt), captured
from a real run on an RTX 2080, sm_75.)
