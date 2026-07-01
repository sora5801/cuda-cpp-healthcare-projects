# 4.19 — Motion-Compensated 4D-CT Reconstruction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.19`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

A chest CT scan takes several seconds, but the patient keeps breathing — so the
anatomy **moves while the scanner spins**, and a plain reconstruction blurs. **4D-CT**
copes by tagging each X-ray projection with the breathing *phase* it was taken in and
binning the projections into a handful of phase groups. Reconstruct each group alone
and you get a stack of images, but each one is **severely under-sampled** (few angles)
and *still* blurred by residual motion. **Motion-compensated reconstruction (MCR)**
fixes this: given a **Deformation Vector Field (DVF)** describing how every point moved
from a chosen reference phase into each other phase, it **warps every projection's
contribution back into the reference frame**, so all phases reconstruct one sharp image.
Motion goes from a curse into free extra angles.

This project is a **reduced-scope, 2-D teaching version**: parallel-beam geometry, a
*known* analytic breathing DVF (not estimated), and a side-by-side comparison of naive
4D-FBP vs motion-compensated reconstruction so you can *watch a moving nodule re-focus*.
The per-pixel reconstruction runs on the GPU (one thread per output pixel) and is
checked bit-for-bit against a plain CPU reference.

## What this computes & why the GPU helps

4D-CT captures respiratory motion by sorting ~4,000 projections into ~10 breathing
phases, then reconstructing each phase — effectively 10 independent 3-D reconstruction
problems with very few (~400) projections each (severe under-sampling). Simultaneous
motion-compensated reconstruction (MCR) jointly estimates the reference volume and the
DVF by alternating image reconstruction with non-rigid registration, each a
GPU-intensive computation. 4D-CBCT for adaptive radiotherapy is harder still (sparser
projections, imaging-dose limits) and relies on GPU-accelerated iterative
reconstruction with motion-model regularization.

**The parallel bottleneck:** the **backprojection gather**. Every output pixel sums an
interpolated sample from *every* (phase, angle) projection, and for MCR it *also*
evaluates the DVF once per phase to know where to look. Pixels are completely
independent, so the whole reconstruction is embarrassingly parallel — one GPU thread
per pixel, exactly the pattern flagship `4.01` uses, plus a per-phase coordinate warp.

## The algorithm in brief

- **Phase binning:** projections are grouped by breathing phase; each phase is sparse.
- **Ramp filter (Ram-Lak):** the "Filtered" in Filtered BackProjection — undoes the
  1/r blur inherent to backprojection. Applied identically to both reconstructions.
- **Naive 4D-FBP:** backproject every projection into the reference grid ignoring
  motion → moving structures smear across their per-phase positions.
- **Motion-compensated backprojection:** before sampling phase *p*'s projections,
  displace each pixel by the phase-*p* DVF, so all phases reconstruct the same
  reference geometry → the moving nodule re-focuses.
- **DVF (motion model):** here a smooth, analytic diaphragm-like breathing warp. In
  production it is *estimated* by deformable image registration (Demons / optical flow),
  often inside an alternating MCR loop.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/motion-compensated-4d-ct-reconstruction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/motion-compensated-4d-ct-reconstruction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\motion-compensated-4d-ct-reconstruction.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (optional CMake build)
```

The demo builds if needed, runs on `data/sample/sinogram4d_sample.txt`, prints the
naive-vs-MCR result, shows the GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/sinogram4d_sample.txt` — a tiny synthetic
  breathing-phantom sinogram (8 phases × 10 angles, 129 detectors, 96×96 image) so the
  demo runs with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` (documented; downloads nothing —
  points at DIR-Lab / TCIA / POPI and never bypasses registration).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: DIR-Lab 4D-CT lung dataset (<https://www.dir-lab.com/>) — 10
cases with expert landmark pairs; TCIA 4D-CT lung radiotherapy collections; POPI model
(<https://www.creatis.insa-lyon.fr/rio/popi-model>); CIRS dynamic lung phantom data.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt). The headline:

```
naive 4D-FBP  peak = 0.8966 ...      <- motion smeared the nodule below its true density
motion-comp   peak = 1.0413 ...      <- motion compensation recovered it toward 1.0
peak recovery (MCR / naive) = 1.1615x   (true nodule density = 1.0)
RESULT: PASS (GPU matches CPU within tol=1.0e-03; MCR recovers the moving nodule)
```

The program reconstructs **naive** and **motion-compensated** images on both the **GPU**
(`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`) and asserts they
agree within `1e-3` (observed ~`2e-6`). Because both call the *same* shared physics in
[`src/mc4dct.h`](src/mc4dct.h), the agreement is essentially machine precision. As a
*second* check on the science, the motion-compensated peak recovers the nodule's known
density (1.0) within a few percent — see PATTERNS.md §4.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the sinogram, ramp-filters, runs CPU + GPU for
   both reconstructions, verifies, and reports the peak-recovery result.
2. [`src/mc4dct.h`](src/mc4dct.h) — **the physics**: the shared `__host__ __device__`
   per-pixel reconstruction (`mc_pixel`), the DVF (`dvf_at`), and the breathing model.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-to-pixel idea.
4. [`src/kernels.cu`](src/kernels.cu) — the kernel (calls `mc_pixel`) and host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — loader, ramp filter, the serial
   baseline (also calls `mc_pixel`), and the peak/sharpness metrics.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, I/O helpers.

## Prior art & further reading

- **RTK** (<https://github.com/RTKConsortium/RTK>) — 4D **ROOSTER** motion-compensated
  reconstruction; study its motion-model regularization and conjugate-gradient loop.
- **ASTRA** (<https://github.com/astra-toolbox/astra-toolbox>) — GPU forward/back
  projection kernels; learn how production projectors handle geometry and interpolation.
- **TIGRE** (<https://github.com/CERN/TIGRE>) — 4D-capable iterative reconstruction;
  a readable CUDA + MATLAB/Python codebase.
- **Plastimatch** (<https://plastimatch.org/>) — deformable registration + 4D dose;
  see how DVFs are estimated and consumed downstream.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Per-output-pixel **gather** with interpolation (PATTERNS.md §1, exemplar `4.01`): a 2-D
thread grid over the image, each thread looping over all (phase, angle) projections and
— for MCR — warping its pixel by the per-phase DVF before sampling. The per-pixel
physics is a shared `__host__ __device__` core (PATTERNS.md §2) so CPU and GPU results
match. Real MCR additionally uses CUDA Demons/optical-flow for DVF estimation, texture
memory for DVF interpolation, and cuFFT for PCA-based motion models — described in
THEORY.md "Where this sits in the real world".

## Exercises

1. **Estimate the DVF instead of prescribing it.** Replace the analytic `dvf_at` with a
   simple per-phase rigid shift found by cross-correlating each phase's sinogram against
   phase 0 — the first step toward real MCR.
2. **Sweep the breathing amplitude.** Regenerate the data with
   `--amp 0.1 … 0.6` and plot naive peak vs amplitude — motion blur should worsen while
   the MCR peak stays near 1.0.
3. **Add a wrong DVF.** Halve the DVF amplitude used in reconstruction (but not in the
   forward model) and watch the peak recovery degrade — motion compensation is only as
   good as its motion model.
4. **Texture-memory interpolation.** Move the projection sampling into a CUDA texture
   object and compare speed/accuracy (production projectors do this).
5. **Go 3-D / cone-beam.** Extend `mc_pixel` to a voxel + a 3-vector DVF and a cone-beam
   ray — the real 4D-CBCT geometry (see THEORY.md).

## Limitations & honesty

- **Reduced scope on purpose.** 2-D parallel-beam, not 3-D cone-beam; a *known* analytic
  DVF, not one estimated by registration; a single non-iterated backprojection, not the
  alternating reconstruct↔register MCR loop of ROOSTER/PICCS.
- **Synthetic data**, labeled synthetic everywhere. The "breathing" is a smooth cosine
  model, not a measured respiratory trace; densities are arbitrary units.
- **Not a benchmark.** Timings are teaching artifacts; the tiny problem is partly
  launch-bound. The GPU's real edge appears at clinical volume sizes and projection counts.
- **Not for clinical use.** This demonstrates the *idea* of motion-compensated
  backprojection — it is not a calibrated or validated reconstruction pipeline.
