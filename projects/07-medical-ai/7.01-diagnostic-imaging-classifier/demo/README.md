# Demo — 7.1 Diagnostic Imaging Classifier

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/imaging_sample.txt` (4 synthetic
   patches + the fixed model weights).
3. **Verify** the GPU forward pass against the CPU reference
   (`src/reference_cpu.cpp`) and print a clear `PASS`/`FAIL`. The two share the
   same `__host__ __device__` math, so the class logits agree **exactly** (tol 0).
4. **Time** the two kernels (CUDA events) and the CPU baseline — a *teaching
   artifact*, not a benchmark claim (on 4 tiny images the GPU is launch-bound and
   slower than the CPU; the edge grows with batch size and resolution).

The program splits its output deliberately (PATTERNS.md §3):

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric verification error (which vary run
  to run), so it is shown but never diffed.

## What you are looking at

The per-image table shows, for each patch: the predicted class (`normal`/`lesion`),
`P(lesion)` from softmax, the ground-truth label, and whether the prediction was
correct. Then a batch-accuracy line. The two blob patches should read `lesion`
with `P(lesion) ≈ 1`; the two flat/gradient patches should read `normal`.

## Expected result

```
7.1 -- Diagnostic Imaging Classifier
[reduced-scope teaching CNN inference: conv->relu->maxpool->dense->softmax]
images = 4   geometry = 16x16, 4 filters (3x3), 2 classes

 idx  pred     P(lesion)  truth   ok
   0  lesion     1.0000    lesion  yes
   1  lesion     0.9999    lesion  yes
   2  normal     0.3775    normal  yes
   3  normal     0.3775    normal  yes

accuracy: 4/4 correct (100.0%)
RESULT: PASS (GPU matches CPU exactly; tol=0)
```

The `[verify]` line on stderr reports `max |logit_cpu - logit_gpu| = 0.000e+00`,
confirming the GPU kernels reproduce the CPU reference bit-for-bit.
