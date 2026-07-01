# Demo — Project 4.4 Deep-Learning MRI/CT Reconstruction

## What this demonstrates

One command builds the project (if needed), runs it on the committed **synthetic**
under-sampled MRI acquisition, and checks the result. It shows the whole point of
learned / unrolled reconstruction in miniature:

- We keep only **39%** of k-space (~2.5× faster "scan").
- A **zero-filled** reconstruction (inverse-transform the measured k-space
  directly) is blurry and aliased — its RMS error vs the ground-truth phantom is
  printed as the "before".
- An **unrolled cascade** (12 stages of *denoise → data-consistency*) sharpens it,
  lowering the RMS error by **~11%** — printed as the "after".
- The **GPU** result is verified to match the **CPU** reference within `1e-3`.

> This is a **reduced-scope teaching version**: the denoiser is a *fixed* Gaussian
> prior, not a trained CNN, and the transform is a direct DFT, not cuFFT. It
> teaches the *structure* that trained methods (E2E-VarNet on fastMRI) share. See
> `../THEORY.md`.

## Run it

```powershell
# Windows (PowerShell), from the project folder:
./demo/run_demo.ps1
```

```bash
# Linux/macOS (uses the optional CMake build):
./demo/run_demo.sh
```

## What you should see

`stdout` (deterministic; diffed against `expected_output.txt`):

```
4.4 -- Deep-Learning MRI/CT Reconstruction
[REDUCED-SCOPE teaching demo: fixed denoiser prior + k-space data consistency,
 unrolled 12 stages. Not a trained network -- see THEORY.md.]
image = 24 x 24, k-space samples kept = 225 / 576 (39.1%)
RMS error vs truth : zero-filled = 0.102342  ->  reconstructed = 0.090973
recon improved zero-filled by 11.1%
recon diagonal samples (8): 0.0774 0.0689 0.5213 0.7610 1.0074 0.8723 0.0934 0.0862
RESULT: PASS (GPU matches CPU within tol=1.0e-03)
```

`stderr` (shown, not diffed — timings vary run to run):

```
[data]   source: ...\data\sample\mri_scan_sample.txt  (24 x 24 image)
[timing] CPU recon: <ms>   GPU stage kernels: <ms>
[timing] teaching artifact only -- the direct O(N^2) DFT dominates; a real pipeline uses cuFFT ...
[verify] max_abs_err(GPU,CPU) = <~8e-06>  (tolerance 1.0e-03)
```

## How the check works

`stdout` carries only **deterministic** values (dimensions, the RMS-before/after,
a 4-decimal fingerprint of 8 diagonal pixels, PASS/FAIL). We print 4 decimals on
the pixels on purpose: the last one or two digits of a 6-decimal print wobble with
the compiler's fused-multiply-add choices (Debug vs Release), which is exactly the
floating-point lesson in `../THEORY.md`. Timings and the exact `max_abs_err` go to
`stderr` so they never break the diff. The program's **exit code** (0 = GPU matches
CPU) is a second gate the runner checks.
