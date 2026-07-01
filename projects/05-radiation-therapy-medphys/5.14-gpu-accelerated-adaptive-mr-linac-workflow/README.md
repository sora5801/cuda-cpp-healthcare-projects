# 5.14 — GPU-Accelerated Adaptive MR-Linac Workflow

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Radiation%20Therapy%20%26%20Medical%20Physics-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 5: Radiation Therapy & Medical Physics · Catalog ID `5.14`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

> **Reduced-scope teaching version.** The full clinical MR-Linac oART workflow is
> a five-stage, multi-GPU research pipeline (NUFFT MRI reconstruction → deformable
> registration → synthetic-CT CNN → 3-D dose recalculation → fluence
> re-optimization). This project implements the **load-bearing middle of that
> chain on a single 2-D slice** — deformable image registration and dose warping —
> which is enough to teach the GPU pattern and the clinical idea end to end. The
> [THEORY.md](THEORY.md) "Where this sits in the real world" section maps every
> simplification back to the production tool.

## Summary

An MR-Linac images the patient *at the moment of treatment*, so a plan built on an
earlier scan may no longer fit today's anatomy (a full bladder, gas in the bowel,
weight loss). **Online adaptive radiotherapy (oART)** re-derives the delivery from
the daily image while the patient lies on the couch. This project demonstrates the
heart of that adaptation on a synthetic 2-D slice: it **registers** a synthetic
"daily" MR to a "planning" MR (a GPU Demons deformable registration), **warps the
planned dose** onto the daily anatomy, and computes the **plan-approval metrics**
(mean dose, D95, target coverage) a physician would check. Every step is a
per-voxel GPU operation; the CPU reference and GPU kernels share the same
per-voxel math so their results match to ~1e-14.

## What this computes & why the GPU helps

MR-Linac (MRL) systems (Elekta Unity, ViewRay MRIdian) combine MRI with simultaneous radiation delivery, enabling online adaptive radiotherapy (oART) where each fraction's plan is re-optimized based on daily anatomy. The oART workflow must complete all steps within a 30–90 minute treatment slot: (1) real-time MRI reconstruction (GPU NUFFT, <1 s), (2) deformable MR-to-MR registration (GPU Demons/VoxelMorph, <30 s), (3) synthetic CT generation (deep learning CT from MRI, GPU CNN, <10 s), (4) GPU dose recalculation on adapted anatomy (<30 s via collapsed-cone or MC), and (5) re-optimization (<2 min). Every step requires GPU; the entire chain is a GPU pipeline.

**The parallel bottleneck:** deformable image registration is the time-critical,
compute-heavy stage that oART cannot skip and that the GPU exists to accelerate.
It sweeps **every voxel, every iteration**: warp the moving image (a bilinear
gather per voxel), compute a per-voxel force from image gradients, and Gaussian-
smooth the displacement field. On a clinical 3-D volume that is hundreds of
millions of independent voxel updates × tens of iterations — perfectly data-
parallel, so one GPU thread per voxel turns a minutes-long serial sweep into
seconds. Dose warping is the same per-voxel gather and rides along for free.

## The algorithm in brief

Real-time MRI reconstruction (radial GRASP GPU), MR-to-MR deformable registration (Demons, SyN), synthetic CT generation (CNN: MR→sCT), GPU collapsed-cone dose on sCT, GPU proton or photon dose recalculation, warm-start IMRT fluence re-optimization, plan approval metric computation.

What this reduced-scope version actually implements:

- **Demons deformable registration** — iterate: (1) backward-warp the moving image
  by the current displacement field (bilinear gather); (2) add Thirion's optical-
  flow force `-(M−F)∇F / (|∇F|² + (M−F)²/K)` at each voxel; (3) Gaussian-smooth the
  field (the elastic/diffusion regulariser). Repeat for `iters` iterations.
- **Dose warp** — apply the recovered field to the planned dose (backward-warp) so
  it lands on the daily anatomy.
- **Plan-approval metrics** — over the GTV mask: mean dose, D95 (dose covering
  ≥95% of target voxels), and coverage fraction above a threshold.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/gpu-accelerated-adaptive-mr-linac-workflow.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/gpu-accelerated-adaptive-mr-linac-workflow.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\gpu-accelerated-adaptive-mr-linac-workflow.sln /p:Configuration=Release /p:Platform=x64
```

No extra CUDA libraries are linked — the kernels are hand-rolled so the learner can
read every operation (only `cudart_static.lib`, the CUDA runtime, is needed).

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/oart_case.txt`, prints the
registration/dose result, shows the GPU-vs-CPU agreement check, and prints a
timing line (stderr).

## Data

- **Sample (committed):** `data/sample/oart_case.txt` — a tiny **synthetic** 2-D
  case (planning MR, daily MR, dose, GTV) so the demo runs offline with zero
  downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print how to obtain real
  MR-guided RT images (they never bypass registration/licensing).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: MR-Linac Consortium shared datasets (verify URL at mrlinac.org); TCIA MR-guided RT datasets; AAPM MR-Linac WG test cases; MRI-only radiotherapy datasets from published cohorts.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt):

```
5.14 -- GPU-Accelerated Adaptive MR-Linac Workflow
[reduced-scope teaching version: 2-D Demons registration + dose warp; synthetic data]
case: 32x32 voxels, 60 Demons iters, sigma=1.50, K=1.00, dose_thresh=30.00 Gy
registration MSE(moving vs fixed): before=0.024537  after=0.000069
peak displacement magnitude: 7.110311 voxels
GTV plan metrics on WARPED dose:  mean=42.056072 Gy  D95=21.902282 Gy  coverage(>=30.00 Gy)=0.786517
RESULT: PASS (GPU matches CPU within tol=1.0e-06)
```

The program runs the workflow on both the **GPU** (`src/kernels.cu`) and a **CPU
reference** (`src/reference_cpu.cpp`) and asserts the displacement field, warped
dose, and metrics all agree within the documented `1e-6` tolerance — that
agreement is the correctness guarantee. The **MSE dropping ~356×** shows the
registration converged; the **restored GTV coverage** shows the adaptation worked.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the case, runs CPU + GPU, verifies, reports.
2. [`src/mrl_registration.h`](src/mrl_registration.h) — the shared `__host__ __device__`
   per-voxel physics (bilinear sample, image gradient, Demons force). **The core.**
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the per-voxel pattern.
4. [`src/kernels.cu`](src/kernels.cu) — the four kernels and the host-driven Demons loop.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

Gadgetron (https://github.com/gadgetron/gadgetron) — real-time GPU MRI reconstruction for MRL; Plastimatch (https://plastimatch.org/) — GPU DIR + sCT generation; matRad (https://github.com/e0404/matRad) — dose re-optimization kernel; MONAI (https://github.com/Project-MONAI/MONAI) — CNN for MR→sCT translation.

What to learn from each:

- **Plastimatch** — the closest match to *this* project: a production GPU
  deformable image registration (Demons and B-spline) and dose-warping engine.
  Read its Demons implementation to see how the same force we hand-roll is
  organized for 3-D volumes and multi-resolution pyramids.
- **Gadgetron** — how the *upstream* MRI reconstruction stage (NUFFT) is turned
  into a streaming GPU pipeline; the stage we omit but assume.
- **matRad** — the *downstream* dose calculation and IMRT fluence re-optimization;
  what "warm-start re-optimization" looks like in a real planning code.
- **MONAI / ITK** — MONAI for the CNN synthetic-CT stage (MR→sCT); ITK's
  `DemonsRegistrationFilter` for the canonical reference implementation of the
  algorithm taught here.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

Per-voxel **gather + stencil**, host-driven iteration. Each kernel maps one GPU
thread to one image voxel over a 2-D grid of 16×16 tiles: warping is a bilinear
**gather** (like CT backprojection, project 4.01), the Demons force and Gaussian
smoother are bounded **stencils** (like the lattice-Boltzmann solver, 6.04), and
the host drives the Demons iteration keeping all state resident on the device
between launches. No atomics anywhere (each output voxel is written by exactly one
thread), so the result is deterministic. The full clinical version adds cuFFT
(NUFFT), cuDNN (sCT CNN), a 3-D dose engine, and cuSPARSE (optimizer) as CUDA-
stream stages — see THEORY.md.

## Exercises

1. **Recover the ground truth.** The synthetic daily image is shifted by exactly
   `(3, 2)` voxels. Add a stat that reports the *mean* displacement inside the GTV
   and check it approaches `(3, 2)`. Then change `--dx/--dy` in `make_synthetic.py`
   and confirm the registration tracks it.
2. **Multi-resolution.** Demons converges slowly on large motions. Add a 2-level
   image pyramid (downsample ×2, register, upsample the field, refine) and compare
   iteration counts for a big shift.
3. **Shared-memory smoothing.** The Gaussian pass re-reads global memory for each
   tap. Tile the input into shared memory with a halo (as in project 7.10) and
   measure the bandwidth win at 64×64 or 128×128.
4. **A better regulariser.** Swap the Gaussian (diffusion) smoothing for a
   fluid/elastic regulariser, or symmetrize the force (use `∇M` too). Discuss the
   effect on invertibility of the deformation.
5. **Inverse-consistency check.** Register M→F and F→M and verify the two fields
   approximately cancel — a standard clinical QA metric for DIR.

## Limitations & honesty

- **Reduced scope.** This is *not* a clinical oART engine. It omits the four other
  pipeline stages (NUFFT reconstruction, synthetic-CT CNN, a real 3-D dose engine,
  fluence re-optimization). It runs on a single 2-D slice, not a 3-D volume.
- **Synthetic data.** The images, dose, and GTV are procedurally generated and
  carry **no clinical meaning**. The "dose" is a Gaussian cloud, not a real
  transport calculation. Nothing here may inform a treatment decision.
- **Simplified registration.** We use single-resolution additive Demons with
  Gaussian (diffusion) regularisation — the readable baseline. Production DIR uses
  multi-resolution pyramids, diffeomorphic/symmetric formulations, and masks to
  guarantee invertible, anatomically plausible deformations. The recovered field
  overshoots in low-signal background regions (visible as the peak displacement
  exceeding the true motion); real codes mask those out.
- **Dose warping ≠ dose recalculation.** Warping the planned dose assumes the
  deformation captures the anatomy change; a real adaptive workflow *recomputes*
  dose on the daily anatomy (collapsed-cone or Monte Carlo) rather than only
  deforming the old distribution.
- **Timing is a teaching artifact.** At 32×32 the workload is launch-bound and the
  GPU may not beat the CPU; the point is the *pattern*, whose advantage grows with
  problem size. Not a benchmark claim (CLAUDE.md §12).
