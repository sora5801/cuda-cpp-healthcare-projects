# Demo — 2.35 Electron Paramagnetic Resonance (EPR/DEER) Constrained Modeling

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed synthetic ensemble `data/sample/deer_sample.txt`.
3. **Back-calculate** each frame's DEER distance distribution `P_m(r)` on both the
   CPU and the GPU, and verify they agree bit-for-bit.
4. **Reweight** the ensemble to the experimental target `P_exp(r)` with the shared
   maximum-entropy (BioEn/EROS) solver, and verify the recovered weights match.
5. **Time** the GPU kernel (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so
  it is shown but never diffed.

## Expected result

```
2.35 -- Electron Paramagnetic Resonance (EPR/DEER) Constrained Modeling
ensemble: 64 frames, 24 rotamers/site, 50-bin P(r) over 1.5-6.5 nm
DEER back-calc: GPU vs CPU per-frame P(r) match = YES
reweighting: 4000 steps, theta = 1.0e-04
  chi^2(uniform)    = 6.138988e-02
  chi^2(reweighted) = 2.790492e-04
  P(r) peak bin: uniform r=3.25 nm | reweighted r=3.45 nm | target r=3.45 nm
  true-frame population: prior 0.2500 -> reweighted 0.9895  (16/64 frames are true matches)
RESULT: PASS (GPU back-calc matches CPU; reweighting recovers the true frames)
```

## How to read it

- **`GPU vs CPU per-frame P(r) match = YES`** — the histogram kernel and the CPU
  reference share the exact same `__host__ __device__` math (`src/deer.h`), so
  every bin agrees to 0.0 (see the stderr `[verify]` line).
- **`chi^2(uniform)` → `chi^2(reweighted)`** — the fit to the experimental target
  improves by ~220× as reweighting concentrates population on the frames that
  reproduce `P_exp(r)`.
- **The P(r) peak** moves from the uniform-average value to the target value
  (3.45 nm) — the reweighted model now *looks like the experiment*.
- **`true-frame population: 0.2500 -> 0.9895`** — the headline result. The 16
  synthetic "true" frames (a quarter of the ensemble) carry ~99% of the
  reweighted population: the method **recovered the known answer** without ever
  being told which frames were true.

The stderr timing shows the small CPU baseline is faster on this tiny 64-frame
sample (the GPU launch dominates); the GPU's advantage grows with ensemble size
and rotamer-library size, as the note explains.
