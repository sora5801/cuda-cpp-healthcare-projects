# Demo — 7.18 Retinal Fundus AI Screening

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed synthetic fundus image `data/sample/fundus_sample.txt`.
3. **Verify** the GPU forward pass against the CPU reference (`reference_cpu.cpp`):
   the logits, softmax probabilities, and Grad-CAM heatmap must agree within
   tolerance, and both must predict the **same DR grade**. Prints `PASS`/`FAIL`.
4. **Time** the two convolution layers (CUDA events) versus the CPU forward pass —
   a *teaching artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## Expected result

```
7.18 -- Retinal Fundus AI Screening
[teaching CNN inference: conv->relu->pool x2 -> GAP -> FC -> softmax]
image: 32x32 RGB  (channel-major, normalized [0,1])
predicted DR grade: 2 moderate
class probabilities: 0.211474 0.201075 0.233411 0.154383 0.199657
Grad-CAM 8x8 peak = 0.225891 at (row=4,col=2)
RESULT: PASS (GPU matches CPU within tol=1.0e-03; same grade)
```

## How to read it

- **predicted DR grade** — the argmax of the 5-way softmax over the diabetic-
  retinopathy severity scale (0 none … 4 proliferative). Because the weights are
  **fixed and untrained**, the specific grade is meaningless clinically — it
  demonstrates the *pipeline*, not a diagnosis (**not for clinical use**).
- **class probabilities** — the softmax vector; it sums to 1.0.
- **Grad-CAM peak** — the location on the coarse 8×8 feature grid where the
  winning class was most strongly supported, a stand-in for lesion localisation
  (`THEORY.md` §6). On this synthetic image it lands near the bright disc / spot.
- **RESULT** — `PASS` iff the GPU and CPU agree within `1e-3` **and** pick the
  same grade. The stderr line shows the actual `max_abs_err` (~`1e-8` here — far
  under tolerance; the tiny gap is float-summation order in the average-pool
  reduction, `THEORY.md` §5).
