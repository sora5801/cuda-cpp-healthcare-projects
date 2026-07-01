# Demo — 4.2 Iterative / Model-Based CT Reconstruction

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/` input.
3. **Verify** the GPU result against the CPU reference (`reference_cpu.cpp`) and
   print a clear `PASS`/`FAIL`.
4. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## Expected result

```
4.2 -- Iterative / Model-Based CT Reconstruction
SIRT+TV: 48 angles x 67 detectors -> 48x48 image, 60 iterations
lambda = 1.500  tv_weight = 0.0100
center pixel value = 1.0091
max reconstructed value = 1.6538 at (px,py)=(24,33)
central row profile (8 samples): -0.0074 0.2525 1.1263 1.0324 1.0009 0.9493 0.7014 0.0141
reconstruction RMSE vs truth = 0.1043
RESULT: PASS (GPU matches CPU within tol=2.0e-03)
```

## What the learner is seeing

- **SIRT+TV** runs 60 iterations of forward projection → residual → backprojection
  update → TV smoothing, entirely on the GPU, then compares to the serial CPU SIRT.
- **`center pixel value ≈ 1.0091`** — the reconstruction recovers the phantom's
  body density (1.0) at the center. The **central row profile** traces the disc:
  low at the edges, ~1.0 across the body, with the inserts perturbing it.
- **`reconstruction RMSE vs truth = 0.1043`** — how close the reconstruction is to
  the *known* synthetic phantom (a science check, printed only because the sample
  ships ground truth). This is separate from the CPU-vs-GPU check.
- **`RESULT: PASS`** — the GPU image matches the CPU reference within `2·10⁻³`.
  (Exact equality is impossible: the GPU's fused multiply-add diverges from the
  host compiler by ~`7·10⁻⁴` over 60 iterations — see THEORY.md §6.)
- **stderr** (shown, not diffed) carries the CPU vs GPU **timing** and the measured
  `max_abs_err`; both vary run to run, which is why they are not in the diff.

To make the effect of the prior visible, regenerate the sample with `--tv 0`
(`python scripts/make_synthetic.py --tv 0`) and re-run: the noise the TV step was
suppressing comes back. See the exercises in the top-level README.
