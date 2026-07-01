# THEORY — 5.14 GPU-Accelerated Adaptive MR-Linac Workflow

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

> **Scope of this document.** The catalog entry describes a full five-stage,
> multi-GPU clinical pipeline. This teaching project implements the **deformable
> registration + dose-warping** middle of that chain on a 2-D slice. Sections
> 1–6 explain what we actually build; section 7 ("Where this sits in the real
> world") explains the full clinical workflow and every simplification we made.

---

## 1. The science

**The clinical problem.** External-beam radiotherapy aims a shaped, high-energy
beam at a tumour while sparing nearby healthy tissue. The plan — how much dose to
deposit where — is normally computed once, days before treatment, on a planning
CT/MR. But anatomy moves. Between fractions (and even within one), a bladder
fills, bowel gas shifts, the patient loses weight, the tumour shrinks. If the beam
is delivered to yesterday's geometry, the tumour can be under-dosed and healthy
organs over-dosed.

**The MR-Linac.** An MR-Linac (Elekta Unity, ViewRay MRIdian) bolts an MRI scanner
onto a linear accelerator, so the machine can *see soft tissue in real time while
it treats*. That enables **online adaptive radiotherapy (oART)**: image the patient
on the couch, and if the anatomy has changed, re-derive the plan for *today's*
geometry before delivering the beam — all within the 30–90 minute slot the patient
is lying there.

**Where registration fits.** The first analytic step after imaging is to figure
out *how* the anatomy moved: a voxel-by-voxel map from the planning image to the
daily image. That map is a **deformable image registration (DIR)**. Once you have
it, you can carry structures (the tumour contour, organs-at-risk) and the planned
dose from the planning frame onto the daily anatomy, and check whether the plan
still covers the target. DIR is the connective tissue of the whole oART chain, and
it is expensive — which is exactly why it runs on the GPU.

This project models a single 2-D MR slice: a **planning image** `F` (fixed), a
**daily image** `M` (moving, the same anatomy shifted), a **planned dose**, and a
**GTV mask** (the tumour). We recover the deformation `M → F`, warp the dose onto
the daily anatomy, and report the plan-approval numbers.

## 2. The math

**Images.** `F, M : Ω → ℝ` are intensity fields on a discrete 2-D grid
`Ω = {0..nx-1} × {0..ny-1}`. A voxel is addressed by `(x, y)`; the flat index is
`i = y·nx + x`.

**Deformation.** We seek a displacement field `φ(x) = (u(x,y), v(x,y))` in *voxel
units* such that the moving image, resampled through `φ`, matches the fixed image:

```
    M(x + φ(x))  ≈  F(x)      for all x ∈ Ω.
```

Sampling `M` at the non-integer location `x + φ(x)` uses **bilinear
interpolation** (`sample_bilinear` in `mrl_registration.h`). This "pull" / backward
convention means each *output* voxel independently reads one *source* location — no
two threads write the same place, so it parallelises without locks.

**Objective.** We minimise the sum-of-squared-differences (SSD) between the warped
moving image and the fixed image:

```
    E(φ) = 1/2 · Σ_x ( M(x + φ(x)) − F(x) )² .
```

**Demons force (Thirion).** The gradient of the per-voxel term w.r.t. the
displacement is `(M − F)·∇M(x+φ)`. Thirion's *demons* algorithm makes the key
approximation `∇M(x+φ) ≈ ∇F(x)` (use the cheap, fixed image gradient), and takes a
normalised gradient-descent step:

```
                     -( M(x+φ) − F(x) ) · ∇F(x)
    δφ(x)  =  ──────────────────────────────────────────
                  |∇F(x)|²  +  ( M(x+φ) − F(x) )² / K
```

- `M(x+φ) − F(x)` is the intensity mismatch after the current warp (unitless).
- `∇F = (∂F/∂x, ∂F/∂y)`, computed by central differences (units: intensity/voxel).
- The minus sign makes it a **descent** step (reduce the mismatch next iteration).
- The denominator is Thirion's normaliser. `|∇F|²` turns the step into (roughly) a
  unit displacement toward the matching isocontour; the `(M−F)²/K` term (with
  `K` ≈ squared voxel spacing, here 1) keeps the step finite where `∇F ≈ 0`
  (flat regions carry no directional information).

**Regularisation (diffusion).** The raw force field is noisy and can tear. After
each iteration we convolve the field with a Gaussian of width `σ`:

```
    φ ← G_σ * (φ + δφ) .
```

Gaussian smoothing is the *diffusion regulariser*: it enforces spatial smoothness,
producing an elastic, near-invertible deformation instead of a jagged one. A 2-D
Gaussian is **separable**, so we do it as a horizontal 1-D pass then a vertical
1-D pass (`gaussian_kernel_1d` builds the weights).

**Dose warp & metrics.** The planned dose `D` (Gray) was defined on `F`; the same
`φ` maps it to the daily frame: `D_daily(x) = D(x + φ(x))`. Over the GTV mask we
report **mean dose**, **D95** (the dose level received by ≥95% of GTV voxels — the
5th percentile of the sorted GTV dose), and **coverage** (fraction of GTV at or
above a threshold).

## 3. The algorithm

```
oart(F, M, D, GTV, iters, σ, K):
    φ ← 0                                 # identity: no deformation yet
    (gfx, gfy) ← ∇F                        # fixed-image gradient, computed ONCE
    for it in 1..iters:
        Mw ← warp(M, φ)                    # bilinear gather, per voxel
        for each voxel x:                  # per voxel, independent
            φ(x) += demons_force(Mw(x), F(x), gfx(x), gfy(x), K)
        φ ← gaussian_smooth(φ, σ)          # separable, per voxel
    D_daily ← warp(D, φ)                   # bilinear gather, per voxel
    metrics ← appraise(D_daily, GTV)       # mean / D95 / coverage
```

**Complexity.** Let `N = nx·ny` voxels, `T = iters`, `R = 3σ` (smoothing radius).

- Per iteration: one warp `O(N)`, one force pass `O(N)`, two separable smoothing
  passes `O(N·R)`. Total serial cost `O(T · N · R)`.
- **Work is fully parallel:** every voxel update in every stage is independent
  (reads only its own value plus a small fixed neighbourhood). The parallel *depth*
  per iteration is `O(1)` for warp/force and `O(log R)`-ish for a smart reduction
  (we use a simple `O(R)` window). Across iterations the depth is `O(T)` because
  each iteration depends on the previous field — that serial chain is why the
  **host** drives the loop and the **device** does each step.
- **Arithmetic intensity** is low (a handful of FLOPs per global-memory read), so
  the kernels are **memory-bandwidth bound** — the classic profile for image
  operators, and the reason coalesced row-major access matters.

## 4. The GPU mapping

**Thread-to-data.** Every kernel uses the same 2-D mapping: a grid of `16×16`
thread blocks tiles the image; thread `(blockIdx.x·16+threadIdx.x,
blockIdx.y·16+threadIdx.y)` owns voxel `(x, y)` and writes output element
`i = y·nx + x` exactly once. Row-major layout means threads in a warp (consecutive
`x`) touch consecutive addresses → **coalesced** global loads/stores.

**Launch configuration.** `block = (16, 16) = 256` threads: a multiple of the
32-lane warp, high occupancy on sm_75…sm_89, and a natural 2-D tile.
`grid = (⌈nx/16⌉, ⌈ny/16⌉)`; edge tiles are guarded by `if (x>=nx || y>=ny) return;`.

**The four kernels** (all in `kernels.cu`):

| Kernel | Pattern | Reads | Writes |
|---|---|---|---|
| `grad_kernel` | stencil (3-pt) | `F` neighbourhood | `gfx, gfy` (once) |
| `warp_kernel` | gather (bilinear) | `src`, `u`, `v` | `dst` |
| `demons_add_kernel` | pointwise | `warped, F, gfx, gfy, u, v` | `u, v` |
| `smooth_axis_kernel` | stencil (1-D window) | `in`, weights `w` | `out` |

**Memory hierarchy.** This teaching version keeps everything in **global memory**
and reads through `__restrict__` pointers (lets the compiler cache in registers
and assume no aliasing). Two deliberate choices worth noting:
- The Gaussian weights `w` (a few doubles) are uploaded once and read by every
  thread — a textbook case for **constant memory** or shared memory; we leave them
  in global and flag the optimisation as Exercise 3, keeping the code minimal.
- The smoothing re-reads neighbours from global memory; a production kernel would
  stage a tile-plus-halo into **shared memory** (as project 7.10 does for 1-D
  convolution). We keep the naive version because it is legible and correct.

**Host-driven iteration, device-resident state.** The outer Demons loop lives on
the host, but the fields `u, v`, images, and gradient **never leave the device
between iterations** — only the final results are copied back. This "host loop,
device kernels, no D2H in the hot path" structure is exactly how a real oART engine
overlaps stages; here it also means the CUDA-event timer measures compute, not PCIe.

```
     2-D image (nx × ny)                 grid of 16×16 blocks
   ┌───────────────────────┐           ┌─────┬─────┬─────┐
   │  voxel (x,y)  ...      │           │ B00 │ B10 │ B20 │   each block = 256
   │  i = y·nx + x          │  ──────►  ├─────┼─────┼─────┤   threads; thread
   │  one thread per voxel  │           │ B01 │ B11 │ ... │   (tx,ty) ↔ voxel
   └───────────────────────┘           └─────┴─────┴─────┘
```

**No CUDA library is used** in this reduced version (the kernels are hand-rolled so
nothing is a black box). The full pipeline would call cuFFT (NUFFT reconstruction),
cuDNN (sCT CNN), and cuSPARSE (fluence optimizer) — see §7.

## 5. Numerical considerations

- **Precision: FP64 (double) throughout.** Registration is iterative and the
  bilinear gather composes rounding across `iters` steps; double keeps the CPU and
  GPU close enough to verify tightly, and the small data size makes the FP64 cost
  irrelevant. A production 3-D engine would use FP32 (or mixed precision) for
  memory/bandwidth and accept a looser tolerance.
- **No atomics, no races.** Every kernel writes each output voxel from exactly one
  thread. The warp/smooth kernels read a *different* buffer than they write
  (ping-pong for smoothing via a scratch buffer), so there is no read-after-write
  hazard. Because there are no floating-point atomics, the summation order is
  fixed and the result is **bitwise-deterministic across runs** (PATTERNS.md §3).
- **Determinism of the report.** All run-to-run varying numbers (timings) go to
  **stderr**; the deterministic result goes to **stdout**, which the demo diffs.
  The D95 uses a plain ascending sort of a fixed multiset of GTV doses → stable.
- **Edge handling.** `sample_bilinear`, `grad_x/y`, and the smoother all clamp
  indices to `[0, n-1]` (clamp-to-edge). This is consistent on CPU and GPU (same
  `clampi`), so borders don't introduce a mismatch.

## 6. How we verify correctness

The GPU workflow (`oart_gpu`) is checked against an independent serial CPU
reference (`oart_cpu` in `reference_cpu.cpp`). Both call the **same** per-voxel
functions from `mrl_registration.h`, so the arithmetic is identical; they differ
only in execution order (nested host loops vs. a grid of threads).

- **What we compare:** the displacement field `u, v`, the warped dose, and the
  scalar metrics (mean GTV dose, D95). We take the max absolute difference over all
  voxels.
- **Tolerance: `1e-6`.** Registration is a *long iterative* solver, so even in
  double precision the GPU's fused multiply-add and the host compiler's arithmetic
  diverge slightly and accumulate over the iterations. In practice we observe
  `~1e-14` on the field and `~2e-14` on the dose — far below the `1e-6` gate — but
  we set the tolerance to a physically-negligible `1e-6` rather than claim
  bit-identity, per PATTERNS.md §4. `1e-6` voxels of displacement (or `1e-6` Gy of
  dose) is meaningless clinically.
- **A stronger, physical check.** Beyond CPU==GPU, the demo reports the **SSD/MSE
  before vs after** registration. On the sample it drops from `0.024537` to
  `0.000069` (~356×) — the registration genuinely aligned the images, not just
  "the two implementations agree on a wrong answer." The restored GTV coverage is a
  second science-level sanity check.
- **Why this is convincing:** two implementations written to different execution
  models, agreeing to machine precision *and* producing a physically sensible MSE
  collapse, is strong evidence the kernels are correct.

## 7. Where this sits in the real world

The clinical oART workflow is a **five-stage GPU pipeline** run under hard time
pressure. This project implements stages 2 (registration) and part of 4 (dose
mapping) on a 2-D slice. The full chain:

1. **Real-time MRI reconstruction** — the daily image arrives as raw k-space, often
   on a non-Cartesian (radial/spiral) trajectory for speed (GRASP). Reconstructing
   it is a **NUFFT** (non-uniform FFT) plus coil combination and compressed-sensing
   iteration — a **cuFFT**-based GPU pipeline (see project 8.03 for the FFT idea,
   and **Gadgetron** for the production streaming reconstructor). We *assume* the
   reconstructed image.
2. **Deformable MR-to-MR registration** — *this project*. Production DIR
   (**Plastimatch**, ITK, or a learned **VoxelMorph**/MONAI CNN) uses
   multi-resolution pyramids, diffeomorphic/symmetric formulations (SyN) to
   guarantee invertibility, organ masks, and 3-D volumes with hundreds of millions
   of voxels. We use single-resolution additive Demons on a 2-D slice.
3. **Synthetic-CT generation** — dose calculation needs electron density, which MR
   doesn't provide directly. A CNN (**MONAI**) translates MR → synthetic CT
   (**cuDNN** convolutions). We skip this entirely (we warp a precomputed dose).
4. **Dose recalculation on the adapted anatomy** — a real workflow *recomputes*
   dose on today's synthetic CT with a **collapsed-cone** or **Monte-Carlo** engine
   (see project 5.01 for the MC idea; **matRad** for the planning code). We instead
   *warp* the planned dose through `φ`, which is a fast approximation valid only for
   small deformations — clinically you would recompute.
5. **Warm-start re-optimization** — the IMRT fluence/aperture weights are re-solved
   from the previous plan as a warm start, a large sparse optimisation
   (**cuSPARSE**). We only *appraise* the (warped) plan; we don't re-optimise.

Each real stage is double-buffered across **CUDA streams**, often spanning
**multiple GPUs**, so acquisition, reconstruction, registration, dose, and
optimisation overlap and the whole adaptation fits the couch-time budget. The
single-slice, single-GPU, no-library version here is the pedagogical core: it shows
*why* every stage is a per-voxel GPU problem and how registration + dose mapping
actually work.

---

## References

- **J.-P. Thirion (1998)**, "Image matching as a diffusion process: an analogy with
  Maxwell's demons," *Medical Image Analysis* — the original demons algorithm.
- **Vercauteren et al. (2009)**, "Diffeomorphic demons," *NeuroImage* — the modern
  invertible variant that clinical DIR builds on.
- **Plastimatch** (https://plastimatch.org/) — GPU deformable registration + dose
  warping; the closest production analogue to this project. Read its Demons and
  B-spline registration for the 3-D, multi-resolution version of §2–§4.
- **Gadgetron** (https://github.com/gadgetron/gadgetron) — real-time GPU MRI
  reconstruction (stage 1).
- **matRad** (https://github.com/e0404/matRad) — dose calculation + IMRT
  optimization (stages 4–5).
- **MONAI** (https://github.com/Project-MONAI/MONAI) — CNNs for MR→sCT (stage 3)
  and learned registration (VoxelMorph). ITK's `DemonsRegistrationFilter` is the
  canonical reference implementation of the algorithm taught here.
- **Winkel et al. (2019)**, "Adaptive radiotherapy: The Elekta Unity MR-linac
  approach," *Clinical and Translational Radiation Oncology* — the clinical oART
  workflow and its timing budget.
