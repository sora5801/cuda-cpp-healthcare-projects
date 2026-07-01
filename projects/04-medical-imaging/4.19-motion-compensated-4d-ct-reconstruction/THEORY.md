# THEORY — 4.19 Motion-Compensated 4D-CT Reconstruction

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

A CT scanner rotates an X-ray source/detector around the patient, recording a
**projection** (a set of line integrals through the body) at each angle. Classical
reconstruction assumes the patient is a *static* object during the scan. But a chest
scan takes seconds, and the patient **breathes**: the diaphragm moves the lungs, heart,
and any tumor by up to a couple of centimeters. Reconstruct as if nothing moved and the
moving structures **blur** — a tumor can appear larger, fainter, and mislocated, which
matters enormously for radiotherapy planning (you irradiate where the tumor *is*, not a
smeared ghost).

**4D-CT** ("4D" = 3-D + time) tackles this. A respiratory signal (a chest belt, or a
surrogate extracted from the data) tags every projection with the **breathing phase**
it was taken in — say 0% (full inhale) to ~100% and back. Projections are **binned** by
phase into P groups (typically P ≈ 10). Two problems remain:

1. **Under-sampling.** Splitting ~4,000 projections into 10 bins leaves each bin with
   only ~400 angles — far too few for an artifact-free image. A single-phase image is
   streaky.
2. **Residual motion.** Even within a phase bin there is some motion, and more
   importantly the *anatomy differs between bins*, so you cannot simply pool them.

**Motion-compensated reconstruction (MCR)** solves both at once. Pick one phase as the
**reference** (say end-inhale). Estimate, for every other phase *p*, a **Deformation
Vector Field (DVF)** `u_p(x)` that says how each point `x` of the reference anatomy
moved during phase *p*. Then reconstruct the reference image using **all** projections
from **all** phases, but warp each phase's contribution back through its DVF. Every
projection now informs the reference frame, so the effective angular sampling is the
*full* set — sharp — and motion blur is removed because each ray is placed where its
tissue actually was.

This project isolates the **motion-compensated backprojection** step (the heart of MCR)
in a clean 2-D setting so you can see it work.

---

## 2. The math

### 2.1 Projection (the forward model)

In 2-D parallel-beam geometry, a projection at angle θ is the **Radon transform** of the
attenuation image `f(x, y)`:

```
p_θ(s) = ∫ f( s·cosθ − t·sinθ ,  s·sinθ + t·cosθ ) dt
```

where `s` is the signed detector offset and `t` runs along the ray. Discretely, angle
index `k` has `θ_k = k·π/K` (a half-turn suffices for parallel beam), and detector bin
`j` sits at `s_j = (j − (n_det−1)/2)·ds`.

### 2.2 Filtered BackProjection (the static inverse)

The Fourier-slice theorem gives the inverse. Filtered BackProjection (FBP):

```
f(x, y) = ∫₀^π  ( p_θ ∗ h )( x·cosθ + y·sinθ )  dθ
```

- `h` is the **ramp filter** (`|ω|` in frequency; the discrete **Ram-Lak** kernel in
  space). Without it, backprojection alone gives a `1/r`-blurred image.
- The inner term `x·cosθ + y·sinθ` is the detector offset the ray through `(x,y)` hits at
  angle θ — so backprojection **smears each filtered projection back along its rays**.

### 2.3 Adding motion: the DVF and motion-compensated FBP

Let phase *p* have a DVF `u_p : reference → phase-p`. A reference point `x` sits at
`x + u_p(x)` during phase *p*. So when we backproject a phase-*p* projection into
reference pixel `x`, we must sample it at the **deformed** position:

```
f_ref(x) = (π/K) · Σ_p Σ_{a∈phase p}  ( p_{θ_a} ∗ h )(  (x + u_p(x)) · d(θ_a)  )
```

with `d(θ) = (cosθ, sinθ)`. Set `u_p ≡ 0` and this collapses to naive 4D-FBP (all phases
piled onto the same pixel → motion blur). Use the true `u_p` and every phase lands in the
reference frame → sharp. **That single extra term `u_p(x)` is the whole idea**, and in
the code it is exactly the `if (motion_comp) { … dvf_at … }` branch of
[`mc_pixel`](src/mc4dct.h).

### 2.4 The breathing model used here (analytic DVF)

We prescribe a smooth, physically-plausible field instead of estimating it:

```
m(p) = ½(1 − cos(2π p / P))          breathing amount, m(0)=0, m(P/2)=1
u_p(x) = ( 0.25·A·m(p)·nx ,  A·m(p)·(0.5 − 0.5·ny) )
```

where `(nx, ny) = (x, y)/W` are normalized coordinates and `A` is the amplitude. The
`y`-component is a diaphragm-like vertical push, largest at the bottom of the field; the
small `x`-component makes the field genuinely **non-rigid** (an expansion, not a mere
translation). Phase 0 has zero motion and *is* the reference — consistent with the
reconstruction, which recovers phase 0.

---

## 3. The algorithm

```
1. Load the phase-binned sinogram + geometry.                     (host)
2. Ramp-filter every projection row (Ram-Lak, direct conv).       (host, shared)
3. For motion_comp ∈ {0, 1}:
     for every output pixel (px, py):
        x ← world coords of pixel
        acc ← 0
        for each phase p:
           x' ← x + (motion_comp ? u_p(x) : 0)     # DVF warp
           for each angle a in phase p:
              s ← x'·cos θ + y'·sin θ               # project onto detector
              acc += interp(filtered[p,a], s)       # linear interpolation
        image[px,py] ← acc · (π / K)
4. Verify GPU image == CPU image; report peak recovery.
```

**Complexity.** For an `N×N` image and `K` total angles the backprojection is
`O(N² · K)` work. Serial, that is the `reconstruct_cpu` triple loop. The ramp filter is
`O(K · n_det²)` here (direct convolution; production uses FFT → `O(K · n_det log n_det)`).
DVF evaluation adds `O(N² · P)` cheap arithmetic — negligible next to the `O(N²·K)`
gather.

**Access pattern.** Each output pixel *gathers* one interpolated sample per projection.
Reads are scattered across the filtered sinogram (the detector index depends on the
pixel and angle), writes are one per pixel. This is bandwidth-bound, not compute-bound —
exactly what GPUs excel at when the gather is coherent across neighboring pixels.

---

## 4. The GPU mapping

**Thread-to-data map.** One GPU thread per **output pixel**. A 2-D grid of `16×16`
blocks tiles the `N×N` image:

```
px = blockIdx.x·blockDim.x + threadIdx.x
py = blockIdx.y·blockDim.y + threadIdx.y
thread (px,py) owns image[py·N + px]
```

```
        image  (N x N)                     grid of 16x16 blocks
   +------------------------+         +------+------+------+ ...
   |                        |         | blk  | blk  | blk  |
   |   thread (px,py)  -->  o         |(0,0) |(1,0) |(2,0) |
   |                        |         +------+------+------+ ...
   |                        |         | blk  | blk  | ...
   +------------------------+         +------+------+ ...
   each thread loops over all (phase, angle) projections,
   warps its pixel by the phase DVF, samples, accumulates.
```

**Launch config.** `block = 16×16 = 256` threads: a multiple of the 32-lane warp, 8
warps per block to hide memory latency, and a square tile so neighboring threads read
nearby detector bins (coherent gather). `grid = ceil(N/16) × ceil(N/16)`; the kernel
guards the ragged edge tiles with `if (px>=N || py>=N) return;`.

**Memory hierarchy.**
- **Global memory:** the filtered sinogram and the `cos/sin` tables. The gather from the
  sinogram is the bandwidth bottleneck; texture memory would cache it and do the linear
  interpolation in hardware (a documented exercise / the production choice).
- **Registers:** the per-thread accumulator, world coordinates, and the DVF result — all
  private, no shared memory or atomics needed because each pixel's reduction is
  independent.
- **Constant/`by-value`:** the tiny `Geom` struct is passed by value, so every thread has
  the geometry in registers/constant space — no pointer chasing for scalars.

**No black boxes.** This teaching version hand-rolls the gather so the mapping is visible.
Production MCR pulls in libraries: **cuFFT** for FFT-domain ramp filtering and for
PCA-based motion models; **texture units** for DVF and projection interpolation; and a
**Demons / optical-flow** kernel to *estimate* the DVF — see §7. We keep them explicit
here and name where each would slot in.

---

## 5. Numerical considerations

- **Precision.** The image and sinogram are FP32 (imaging data is ~12–16 bits; FP32 is
  ample). The **trig and floor are done in FP64 then cast** (`cosf_portable`,
  `floorf_portable` in `mc4dct.h`) so the host compiler and nvcc agree bit-for-bit — the
  fast single-precision intrinsics (`__cosf`) would diverge and break CPU==GPU parity.
- **No atomics, no reordering.** Each output pixel is written by exactly one thread and
  its inner sum runs in a fixed order, so there is **no floating-point reduction race** —
  the result is deterministic run-to-run (verified: two runs are byte-identical stdout).
  This is why we can compare CPU and GPU to a *tight* tolerance instead of a loose one.
- **Interpolation & edges.** Rays that miss the detector contribute 0 (guarded in
  `sample_projection`). Linear interpolation between the two nearest bins matches on both
  sides exactly.
- **The DVF is a rigid per-disc shift in the forward model** but a smooth per-pixel field
  in the reconstruction; for small structures under a smooth field these agree well, and
  the ~4% residual in the recovered peak is honest (a finite-resolution / warp-linearity
  effect, not a bug).

---

## 6. How we verify correctness

Two independent checks (PATTERNS.md §4):

1. **CPU == GPU (implementation check).** `reconstruct_cpu` (a plain triple loop) and
   `reconstruct_kernel` (one thread per pixel) call the **same** `mc_pixel` from
   `mc4dct.h`, so they should agree to rounding. We assert `max_abs_err ≤ 1e-3` for
   *both* the naive and the motion-compensated image; observed error is ~`2e-6` (near
   FP32 epsilon over an ~80-term sum). If the kernel had a mapping bug (wrong index,
   missing guard) this would blow up immediately.
2. **Physics check (does motion compensation actually help?).** The moving nodule has a
   known true density (1.0). Motion smears its energy, so the **naive peak** sits below
   1.0 (~0.90); motion compensation re-concentrates it, so the **MCR peak** recovers
   toward 1.0 (~1.04) and its location shifts to the reference-frame position. The
   demo's `PASS` requires *both* CPU==GPU *and* `peak_MCR > peak_naive`. This validates
   the science, not just the plumbing.

Edge cases exercised by the tiny sample: rays that miss the detector, the ragged final
thread-block, and phase 0 (zero motion, where naive and MCR must be identical for that
phase's contribution).

---

## 7. Where this sits in the real world

This is a deliberately **reduced-scope teaching version** (CLAUDE.md §13). Production 4D
reconstruction differs in scale and sophistication:

- **3-D cone-beam, not 2-D parallel-beam.** Real scanners use the **FDK** algorithm (the
  cone-beam extension of FBP; flagship `4.01` builds the parallel-beam core). MCR extends
  FDK per phase (**McKinnon–Bates 4D-FDK**).
- **The DVF is *estimated*, not given.** This is the crux of real MCR. Deformable image
  registration (**Demons**, optical flow, B-splines) estimates `u_p` between phases —
  often **on the GPU** (CUDA Demons). Simultaneous MCR (**PICCS**, **ROOSTER** in RTK)
  *alternates* between (a) reconstructing the reference image and (b) re-estimating the
  DVF, iterating to a joint solution — each half a GPU-heavy computation.
- **Iterative, regularized reconstruction.** Instead of a single backprojection,
  production uses **SART/SIRT** or total-variation-regularized iterations (spatial +
  temporal TV) to fight the under-sampling, especially in sparse **4D-CBCT** for adaptive
  radiotherapy where imaging dose is constrained.
- **Motion models.** A **PCA-based respiratory surrogate** compresses the DVFs to a few
  components (built with **cuFFT/cuSOLVER**), regularizing the estimate and enabling
  real-time gating.
- **Learned priors.** Recent work (score-based diffusion priors, **4D Gaussian
  splatting**, 4D neural radiance fields) pushes sparse 4D-CBCT quality toward 4D-CT using
  GPU-trained models.

What this project keeps faithful: the **motion-compensated backprojection gather** — the
inner computation every one of those systems performs millions of times — and its exact
GPU thread mapping.

---

## References

- **RTK / ROOSTER** — <https://github.com/RTKConsortium/RTK>. The reference open-source
  4D MCR; read the conjugate-gradient + spatiotemporal-TV loop and the warp operators.
- **ASTRA Toolbox** — <https://github.com/astra-toolbox/astra-toolbox>. Production GPU
  forward/back projectors; learn geometry handling and texture interpolation.
- **TIGRE** — <https://github.com/CERN/TIGRE>. Readable CUDA iterative reconstruction,
  4D-capable; a good next step after this project.
- **Plastimatch** — <https://plastimatch.org/>. Deformable registration + 4D dose; how
  DVFs are estimated and consumed.
- Kak & Slaney, *Principles of Computerized Tomographic Imaging* — the FBP/Radon bible.
- McKinnon & Bates (1981), *Towards imaging the beating heart usefully with a conventional
  CT scanner* — the original motion-compensated backprojection idea.
- Rit et al., *The Reconstruction Toolkit (RTK)* — the ROOSTER 4D MCR paper.
