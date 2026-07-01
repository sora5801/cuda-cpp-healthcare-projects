# Demo — 6.8 Tumor Growth & Treatment-Response Modeling

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/tumor_params.txt` (128×128 grid, 400 timesteps,
   a 10 × 2 Gy radiotherapy course).
3. **Verify** the GPU result against the CPU reference (`reference_cpu.cpp`) for
   **both** the treated and the untreated-control scenarios, printing `PASS`/`FAIL`.
4. **Report** tumor-burden and treatment-response metrics, plus a density profile
   across the tumor front.
5. **Time** the CPU baseline and the GPU loop (CUDA events) — a *teaching
   artifact*, not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries timing and the numeric error (which vary run to run), so it
  is shown but never diffed.

## Canonical output

See [`expected_output.txt`](expected_output.txt). From a 1 mm seed, the Fisher-KPP
front grows into a solid core surrounded by an infiltrating rim. The **untreated
control** reaches a larger burden; the **treated** run (10 × 2 Gy, each fraction's
LQ surviving fraction `S ≈ 0.70`) ends with a measurably smaller burden, and the
report quotes the percent reduction and the biologically-effective dose (BED).
`RESULT: PASS` means the GPU and CPU density fields agree to ≤ `1e-6` in **both**
scenarios.

> This is a deliberately **reduced-scope teaching model** (a single normalized
> density field; LQ cell-kill applied as an instantaneous per-cell multiply). It
> is **synthetic and not for clinical use** — it demonstrates the reaction-diffusion
> stencil and linear-quadratic radiobiology, not a calibrated patient tumor.
