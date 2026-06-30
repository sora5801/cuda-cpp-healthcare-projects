# Demo — 2.12 Flexible Fitting / MDFF

## What this demonstrates

Running `run_demo.ps1` (Windows) or `run_demo.sh` (Linux/CMake) will:

1. **Build** the project if the executable is missing.
2. **Run** it on `data/sample/mdff_problem.txt` (a 27-atom model misfitted from
   its target inside a 32³ density map).
3. **Verify** that the GPU MDFF fit and the CPU reference reach the same final
   atom positions (within a physically-negligible tolerance), printing `PASS`/`FAIL`.
4. **Report** the fit quality: RMSD-to-target and density cross-correlation,
   **before vs after**, so you watch the model snap onto the density.
5. **Time** the kernel (CUDA events) and the CPU baseline — a *teaching artifact*,
   not a benchmark claim.

The program splits its output deliberately:

- **stdout** is byte-for-byte deterministic and is diffed against
  [`expected_output.txt`](expected_output.txt).
- **stderr** carries the timing and the GPU-vs-CPU error (which vary run to run),
  so it is shown but never diffed.

## Canonical output

See [`expected_output.txt`](expected_output.txt). The headline numbers are the
**RMSD start → final** (drops as the model fits onto the density) and the
**cross-correlation start → final** (rises as atoms climb onto the density
ridges). `RESULT: PASS` means the GPU and CPU fitted models agree to within
`1e-4` on coordinates of magnitude ~10.

> **Numerical note:** on this machine the GPU and CPU agree to ~`1e-15` (the
> shared `__host__ __device__` math compiles nearly identically here). We still
> verify to a deliberately loose `1e-4` rather than bit-exactness, because the
> GPU's fused-multiply-add (FMA) contraction *can* diverge from the host
> compiler's by ~`1e-6` over hundreds of iterations on other toolchains/cards — a
> real GPU-reproducibility lesson (see THEORY "Numerical considerations"). Don't
> assume bit-identical floating point across GPUs.

> The model is a **synthetic lattice** and the density a simple Gaussian sum —
> a demonstration of the MDFF / trilinear-gather GPU pattern, **not** a validated
> structure-determination pipeline and not for clinical use.
