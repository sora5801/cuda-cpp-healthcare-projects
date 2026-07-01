# Demo — 5.8 Linac QA & Machine Performance Assessment

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on the committed `data/sample/qa_planes_sample.txt` (a synthetic
   planned + measured dose-plane pair).
3. **Compute** the 2-D **gamma-index** map on both the GPU (`kernels.cu`) and a CPU
   reference (`reference_cpu.cpp`), plus the beam **flatness / symmetry / output**
   metrics.
4. **Verify** that the GPU gamma map equals the CPU map **exactly**
   (`max_abs_err = 0`) and that the integer pass counts agree — printing `PASS`/`FAIL`.
5. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the numeric error (which vary run to run), so it
  is shown but never diffed.

## Expected result

```
5.8 -- Linac QA & Machine Performance Assessment
plane: 24 x 24 px  spacing=2.00 mm  norm_dose=100.00
gamma criteria: 3.0% / 3.0 mm  (search radius=5 px, low-dose cut=10%)
gamma pass rate = 100.00%  (396/396 evaluated points pass, gamma<=1.0)
worst gamma = 0.3333 at pixel (4,4)
TG-218 action limit (>=95%): MEETS
machine QA (measured plane, central axis):
  CAX output   = 100.98
  field width  = 38.00 mm (FWHM)
  flatness     = 0.990 %
  symmetry     = 1.961 %
RESULT: PASS (GPU gamma map matches CPU exactly, tol=0)
```

## How to read it

- **gamma pass rate = 100%** — every evaluated pixel has γ ≤ 1, so the (synthetic)
  machine passes. The `worst gamma = 0.333` sits at a **beam-edge** pixel `(4,4)`,
  exactly where the penumbra makes agreement hardest — a good sanity check.
- **TG-218 MEETS** — 100% ≥ the 95% action limit for per-beam IMRT QA.
- **symmetry ≈ 1.96%** and **flatness ≈ 0.99%** — these **recover the error we
  injected** into the synthetic measured plane (2% right-side boost, 1% low output).
  That the analysis reads back the known ground truth is the point of the demo.
- **RESULT: PASS, tol=0** — the GPU and CPU gamma maps are bit-identical because both
  call the same `__host__ __device__` math in `src/gamma.h`.

To see a *failing* machine, edit `scripts/make_synthetic.py` to inject a bigger
error (e.g. output `0.90`), regenerate the sample, rebuild, and re-run — the pass
rate will drop below 95% and `TG-218` will read `BELOW`.
