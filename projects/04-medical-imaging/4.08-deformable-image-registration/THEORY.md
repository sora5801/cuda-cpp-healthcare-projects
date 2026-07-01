# THEORY — 4.8 Deformable Image Registration

> The deep didactic explanation (the "why"). Written for a sharp student who knows C++ but is new to CUDA
> and new to this domain. See [README.md](README.md) for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

Medical images of the *same* patient rarely line up pixel-for-pixel. A lung is a different shape at inhale
than at exhale. A brain shifts and swells between an MRI today and one next month. A tumor and the organs
around it move between the planning CT and the daily cone-beam CT used to aim radiotherapy. To *compare*,
*combine*, or *track* these images, we first have to **warp one onto the other** so that corresponding
anatomy sits at the same coordinates.

**Rigid** registration (one rotation + translation for the whole image) is not enough: tissue *deforms*, and
different parts move differently. **Deformable** (a.k.a. *non-rigid*) registration gives **every pixel its own
displacement vector**. The collection of all those vectors is the **displacement vector field (DVF)**, and
recovering it is what this project does.

Concrete uses:

- **Radiotherapy** — map the dose planned on one CT onto the anatomy of another day; accumulate dose across a
  treatment course as the patient's anatomy changes.
- **Longitudinal studies** — align a patient's scans over time to measure atrophy or tumor growth.
- **Atlas / multi-atlas segmentation** — warp a labeled template brain onto a new patient to transfer labels.
- **Motion modeling** — build a 4-D (breathing) model of the lung from a 4-D-CT.

We model images as smooth intensity functions on a grid and search for the DVF that makes the **moving** image
look like the **fixed** image after warping — the classic image-similarity-plus-smoothness formulation below.

## 2. The math

**Inputs.** A **fixed** image `F : Ω → ℝ` and a **moving** image `M : Ω → ℝ`, both sampled on the same 2-D
grid `Ω = {0..nx−1} × {0..ny−1}`. Intensities here are in `[0,1]` (grayscale). We seek a **displacement
field** `u : Ω → ℝ²`, `u(x) = (u_x, u_y)`, in units of **pixels**.

**Warped moving image.** `M_u(x) = M(x + u(x))`, evaluated by bilinear interpolation (fractional coordinates).

**Objective.** Deformable registration minimizes an energy that trades *similarity* against *smoothness*:

```
E(u) = D( F , M_u )        +   λ · R(u)
        \___ dissimilarity      \__ regularization (smoothness prior)
```

We use **sum of squared differences** for the data term (appropriate when the two images share an intensity
scale, as ours do):

```
D(F, M_u) = Σ_x ( F(x) − M(x + u(x)) )²
```

and a **diffusion / Gaussian** regularizer for `R` (penalizes rough fields).

**Thirion's Demons force.** Rather than form and invert a huge Hessian, Demons takes an *optical-flow*-style
step. Linearizing `M(x+u) ≈ M(x) + ∇M·δu` and assuming near convergence `∇M ≈ ∇F`, the per-pixel update that
descends the SSD is

```
        ( F(x) − M_u(x) ) · ∇F(x)
δu(x) = --------------------------------          (Thirion 1998)
         |∇F(x)|²  +  ( F(x) − M_u(x) )²  +  ε
```

Two things to notice, both of which the code comments call out:

- **The numerator sign is `F − M_u`, not `M_u − F`.** We are *descending* `(M_u − F)²`; the gradient of that
  energy carries a minus sign that flips the difference. Use the wrong sign and every step *climbs* the SSD —
  the field runs away (we hit exactly this bug during development; see `src/demons.h`, `dm_demons_force`).
- **The denominator is a normalization**, not a physical quantity. It makes the step *adaptive*: full-length
  where the gradient is strong and the mismatch small, damped where `∇F ≈ 0` (a flat region gives no reliable
  direction) or where the mismatch is huge (avoid overshoot). `ε` (here `1e-6`) prevents `0/0` in flat,
  already-matched regions.

**Regularization as smoothing.** Instead of adding `λR(u)` to the energy explicitly, *diffusion* Demons applies
the smoothness prior by **convolving the whole field with a Gaussian** `G_σ` after each force step:

```
u  ←  G_σ  *  ( u + δu )
```

This is the fluid/diffusion-regularization trick: a Gaussian blur *is* the Green's function of the diffusion
operator, so smoothing the field is equivalent to a diffusion regularizer with strength set by `σ` (here
1.5 px). Larger `σ` → stiffer, smoother deformation.

## 3. The algorithm

```
u ← 0                                  # identity map: no displacement yet
repeat iters times:
    for each pixel x:                  # (A) FORCE  — O(1) per pixel
        M_u  ← bilinear(M, x + u(x))
        δu   ← Thirion_force(F(x), M_u, ∇F(x))
        u(x) ← u(x) + δu
    u ← GaussianBlur_x(u)              # (B) SMOOTH — separable, O(radius) per pixel
    u ← GaussianBlur_y(u)             # (C)
return u
```

**Complexity.** Let `N = nx·ny` pixels, `T` iterations, `r` the Gaussian half-width.

| Pass | Work per iteration | Notes |
|------|--------------------|-------|
| Force | `O(N)` | 4 interpolation reads + a gradient (4 reads) per pixel |
| Smooth-X | `O(N·r)` | separable: `2r+1` reads per pixel |
| Smooth-Y | `O(N·r)` | separable |

Total serial cost `O(T·N·r)`. Naively convolving with a full 2-D Gaussian would be `O(T·N·r²)`; **separability**
(a 2-D Gaussian factorizes as `G_σ(x)·G_σ(y)`) turns the `r²` into `2r` — the first optimization any stencil
code reaches for. Crucially, **within each pass the pixels are independent**: the force at `x` reads only
`u(x)` (plus the static images), and each smoothing output reads a fixed neighbourhood of the *input* buffer.
That independence is the whole reason a GPU helps.

## 4. The algorithm on the GPU — GPU mapping

Each of the three passes becomes **one kernel, one thread per pixel**, launched over a 2-D grid of 2-D blocks:

```
block = 16 × 16 threads              (= 256 threads, 8 warps — good latency hiding on sm_75..sm_89)
grid  = ceil(nx/16) × ceil(ny/16)    (covers every pixel; edge threads guard x>=nx || y>=ny)

thread (blockIdx, threadIdx) owns pixel:
    x = blockIdx.x*blockDim.x + threadIdx.x
    y = blockIdx.y*blockDim.y + threadIdx.y
```

```
             nx
     +----+----+----+----+ ...
     | B  | B  | B  | B  |     each B = one 16x16 block
 ny  +----+----+----+----+     each cell in B = one thread = one pixel
     | B  | B  | B  | B  |     force / gauss-x / gauss-y all use THIS decomposition
     +----+----+----+----+
```

**The three kernels and their memory:**

1. `demons_force_kernel` — reads `F, M, u`; writes `u[i] += δu`. Pure **gather** (the bilinear warp) + a
   central-difference gradient. **No atomics, no shared memory**: thread `i` writes only its own `u[i]`, so
   the update is race-free by construction.
2. `gauss_x_kernel` / `gauss_y_kernel` — separable **stencil**. Each reads a `(2r+1)` window of a `src`
   buffer and writes one `dst` pixel. `src` and `dst` **must be different buffers** — hence ping-pong.

**Ping-pong buffering.** Two buffers per component (`d_ux`/`d_ux2`, `d_uy`/`d_uy2`):

```
force:     d_ux (in place)                # writes u += du into d_ux
smooth-x:  d_ux  ->  d_ux2                 # blur along x
smooth-y:  d_ux2 ->  d_ux                  # blur along y, lands back "home"
```

After the Y pass the smoothed field is back in `d_ux`/`d_uy`, ready for the next force pass. The images and DVF
stay **resident on the device for the entire loop** — no host↔device copies between iterations — which is
exactly why GPU DIR wins: the only PCIe traffic is one upload of `F,M` and one download of the final field.

**Memory hierarchy used and why.** Everything lives in **global memory** (the images and the two DVF buffers).
`DemonsParams` is passed **by value** into every kernel, so it lands in the driver's *parameter/constant*
space and is broadcast to all threads for free. We deliberately **do not** use shared memory in this teaching
version — the naive global-memory stencil is easier to read, and the shared-memory tiled Gaussian (which would
cut the redundant neighbourhood reads) is left as Exercise 1. The kernels are **bandwidth-bound**, not
compute-bound: each pixel does only a handful of FLOPs per global read, so the win comes from the GPU's memory
parallelism, not its arithmetic throughput.

**Which CUDA library does what — and what hand-rolling costs.** This reduced-scope Demons uses **no** CUDA
math library: the warp, gradient, force, and separable Gaussian are all short hand-written kernels (that is the
teaching point — you can *see* the trilinear-warp and stencil idioms the catalog names). The heavier DIR
variants do lean on libraries: **LDDMM geodesic shooting** applies its smoothing/momentum operator in the
Fourier domain, which is a batched 3-D **cuFFT** forward/inverse per iteration; a Gauss–Newton **B-spline FFD**
assembles and solves a regularization Hessian with **cuBLAS/cuSOLVER**. We describe those in §7 rather than
pull them in, keeping this project a clean "no black boxes" read.

## 5. Numerical considerations

- **Precision: FP64 (double) throughout.** DIR runs *hundreds* of iterations, each adding a tiny `δu` and then
  re-smoothing. Errors accumulate, and in single precision the force denominator (`|∇F|² + diff²`) can lose
  significance where the gradient is small. Double keeps the CPU and GPU in lock-step to ~1e-14 and makes the
  reported SSD stable to many digits. A production 3-D solver often uses FP32 for memory/bandwidth and accepts
  the extra drift — a real trade-off worth knowing.
- **No atomics, deterministic by construction.** Every kernel writes a distinct output pixel, so there is *no*
  floating-point reduction whose order could vary between runs. The program's **stdout is byte-identical every
  run** (verified), which is what lets `demo/run_demo` diff it. (Timings, which do vary, go to **stderr**.)
- **Where CPU and GPU still differ.** They run the *same* `demons.h` formulas, but the GPU fuses
  multiply-adds (FMA) differently than the host compiler. Over 120 iterations this leaves a residual of
  ~5e-15 px — real, tiny, and the reason we verify to a physical tolerance rather than claim bit-identity
  (PATTERNS.md §4).
- **Stability.** The Thirion normalization bounds each step, and the Gaussian smoothing damps high-frequency
  growth; together they keep the explicit iteration stable for the small motions here. Very large motions can
  still fold the field — see §7 (diffeomorphic Demons) and Exercise 5 (Jacobian check).
- **Boundaries.** All reads use **clamp-to-edge**; a zero border would inject a dark ring into the warp and a
  wrapped border would be nonsense for anatomy.

## 6. How we verify correctness

Two independent checks, both in `src/main.cu`:

1. **GPU vs. CPU agreement.** `register_cpu` (plain nested loops, obviously correct) and `register_gpu` (the
   three kernels) call the **same** per-pixel functions in `src/demons.h`, so their displacement fields should
   match to floating-point rounding. We compute the worst per-component difference and require it `≤ 1e-3` px.
   The observed value is ~**5e-15 px** — essentially exact. We pick `1e-3` (not `0`) honestly: it is far below
   any visible motion yet above the FMA drift of a long iterative solver (see §5).
2. **The science actually happened.** Agreement alone would be satisfied even by a *broken* method that moves
   nothing on both sides. So we also report **SSD before vs. after**: on the sample it falls from **51.82 to
   0.064 (a 99.9% reduction)**, and the recovered **mean displacement (4.5 px)** matches the ground-truth
   shift+stretch built into the synthetic pair (~5 px). That is evidence the registration *worked*, not just
   that two implementations agree.

Edge cases exercised by the design: ragged-block guards (image size not a multiple of 16), clamp-to-edge at
all four borders, and flat regions where `ε` prevents `0/0`.

## 7. Where this sits in the real world

This is a **reduced-scope teaching version**: 2-D, single-resolution, SSD, diffusion regularization. Production
DIR differs along several axes, each named in the catalog:

- **3-D and multi-resolution.** Real tools register volumes on a **coarse→fine pyramid** — register downsampled
  images first to capture large motion, then refine. This is both faster and more robust; single-scale Demons
  (ours) can miss motions larger than a structure's own size. (Exercise 2.)
- **Better similarity metrics.** SSD assumes matched intensities. Cross-modality or day-to-day scans need
  **NCC** (local normalized cross-correlation) or **NMI** (mutual information). (Exercise 3.)
- **Diffeomorphic guarantees.** Plain additive Demons can produce a **folding** (non-invertible) field.
  **Diffeomorphic Demons** composes updates via the exponential map, and **SyN** (ANTs) enforces a symmetric,
  invertible transform — the gold standard for brain registration. **LDDMM** goes further, integrating a
  geodesic on the diffeomorphism group (its smoothing operator is applied with **cuFFT** in the Fourier
  domain).
- **B-spline FFD** (Plastimatch, elastix) parameterizes the DVF by a sparse control-point grid instead of a
  dense per-pixel field, and optimizes with Gauss–Newton (a **cuBLAS/cuSOLVER** Hessian solve) — fewer
  parameters, built-in smoothness.
- **Learning-based DIR.** **VoxelMorph** and **TransMorph** train a CNN/transformer to *predict* the DVF in a
  single GPU forward pass (<1 s vs. minutes), trading a one-time training cost for near-instant inference.

Our Demons is the conceptual ancestor of all of these: the warp-force-smooth loop is exactly what the deep
networks learned to shortcut.

---

## References

- **Thirion, J.-P. (1998),** "Image matching as a diffusion process: an analogy with Maxwell's demons,"
  *Medical Image Analysis* — the original Demons force this project implements.
- **Vercauteren et al. (2009),** "Diffeomorphic Demons," *NeuroImage* — the invertible extension (§7).
- **Avants et al. (2008),** "Symmetric diffeomorphic image registration (SyN)," *Medical Image Analysis* —
  the ANTs gold standard used to make ground truth.
- **Balakrishnan et al. (2019),** "VoxelMorph," *IEEE TMI* — the learning-based approach (§7).
- **Plastimatch** (https://plastimatch.org/) — production GPU Demons/B-spline; the closest real sibling to
  study next.
- **DIR-Lab** (https://dir-lab.com/) — 4D-CT lung pairs with expert landmarks; the standard accuracy (TRE)
  benchmark once you move past synthetic data.
