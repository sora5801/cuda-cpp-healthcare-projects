# THEORY — 5.5 Deformable Dose Accumulation & Adaptive Radiotherapy

> The deep didactic explanation (the "why"). Written for a sharp student who knows
> C++ but is new to CUDA and new to this domain. See [README.md](README.md) for the
> quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

A radiotherapy course delivers the prescription dose in ~20–40 daily **fractions**. The plan is designed once,
on a planning CT, to concentrate dose on the tumour (the target) while sparing nearby **organs at risk** (OARs).
But the patient's anatomy is not static across weeks: tumours shrink, bladders and rectums fill and empty,
lungs breathe, and weight changes shift everything. **Adaptive radiotherapy (ART)** responds by re-imaging the
patient at treatment time — cone-beam CT (CBCT) on a conventional linac, or MRI on an MR-Linac — and adapting.

Two questions ART must answer:

1. **What dose did the patient *actually* receive so far?** You cannot simply add each day's dose map voxel by
   voxel, because voxel *(i,j,k)* is a different piece of tissue on different days (the tissue moved). To
   accumulate dose *to the anatomy*, you must first find how the anatomy deformed between days, then move each
   day's dose onto a common reference frame before summing. That common frame is the planning CT.
2. **Should we re-plan?** If the accumulated dose has drifted from the intended plan (target underdosed, an OAR
   over its limit), the clinic re-optimizes the plan on the new anatomy.

This project implements the heart of question 1: **deformable image registration (DIR)** to find the motion,
then **deformable dose accumulation** to sum the moved dose. The clinical stakes are real — mis-estimating
accumulated OAR dose by a few Gray can mean the difference between a safe plan and a spinal-cord or
parotid-gland complication — which is exactly why AAPM **Task Group 132** wrote a QA standard for it.

We ship a **reduced-scope 2-D teaching version**: one planning slice, one daily slice, a synthetic dose cloud,
three identical fractions. The algorithms and the GPU patterns are the real ones; the dimensionality and the
data are scaled down so a learner can read every line and the demo finishes instantly.

## 2. The math

**Images and the displacement field.** Let `F: Ω → ℝ` be the *fixed* (planning) image and `M: Ω → ℝ` the
*moving* (daily) image on a 2-D grid `Ω` of size `nx × ny`. DIR seeks a **displacement vector field (DVF)**
`u(x) = (uₓ(x), u_y(x))` such that the moving image, resampled at the displaced coordinates, matches the fixed
image:

```
F(x) ≈ M(x + u(x))      for all x ∈ Ω.
```

**Thirion's Demons force.** Demons minimizes the sum-of-squared-differences energy `E[u] = Σ_x (M(x+u) − F(x))²`
by a gradient-descent-like update. The per-voxel step is the optical-flow "demons force":

```
                 (F(x) − M(x+u)) · ∇F(x)
   δu(x) = ────────────────────────────────────      (Thirion, 1998)
            |∇F(x)|² + (F(x) − M(x+u))² + ε
```

- `∇F` is the fixed-image gradient (central differences); it is the reliable direction to move along.
- The denominator is Demons' adaptive normalization: the step is long where the gradient is strong and the
  mismatch small, and damped where the gradient vanishes (flat tissue carries no direction) or the mismatch is
  huge (avoid overshoot). `ε` (here `1e-6`) prevents `0/0` in flat, matched regions.
- Sign: `F − M(x+u)` (not `M − F`) makes the step *descend* the SSD energy; the other sign diverges.

After each force step, the field is **regularized** by convolving with a Gaussian `G_σ` (a diffusion prior)
so the deformation stays spatially coherent: `u ← G_σ * (u + δu)`. The Gaussian is *separable*,
`G_σ(x,y) = G_σ(x) · G_σ(y)`, which we exploit on the GPU.

**Deformable dose accumulation.** Let `d_f(x)` be the dose (in Gray) delivered during fraction `f`, defined on
that fraction's grid, and `u_f` the DVF mapping the planning frame to fraction `f`'s anatomy. The
**summation-of-deformed-doses** accumulated dose in the planning frame is

```
   D(x) = Σ_f  d_f( x + u_f(x) ),
```

where each `d_f(x + u_f(x))` is a *warp* (a gather with interpolation). In this demo the three fractions are
identical (`u_f = u`, `d_f = d`), so `D(x) = 3 · d(x + u(x))`. The **dose-volume histogram (DVH)** summarizes
`D`: the (cumulative) DVH at dose level `θ` is the fraction of voxels with `D(x) ≥ θ` — the clinician's headline
readout.

**Objective / I/O.** *Input:* `plan_img, daily_img ∈ [0,1]^N`, `plan_dose, daily_dose ∈ (Gy)^N`. *Output:* the
DVF `u`, the accumulated dose `D` (Gy), and its DVH. *Units:* displacements in voxels (px), dose in Gray.

## 3. The algorithm

```
Stage A — DIR (Thirion Demons), repeat `iters` times:
   for every voxel x:            δu(x) = demons_force(F, M, u, x)       # gather
                                 u(x) += δu(x)
   u ← GaussianX(u)                                                     # stencil
   u ← GaussianY(u)                                                     # stencil

Stage B — deformable dose accumulation:
   for every voxel x:            w(x) = daily_dose(x + u(x))            # gather (warp)
   for f in 1..NFRACTIONS:       D(x) += w(x)                           # accumulate
   for every voxel x:            hist[bin(D(x))] += 1                   # atomic histogram
```

**Complexity.** Stage A is `O(iters · N · radius)` (each Gaussian pass is `O(radius)` per voxel; separability
turns an `O(radius²)` box into two `O(radius)` passes). Stage B is `O(N)` per fraction plus `O(N)` for the
histogram. Serial depth is therefore `O(iters · N · radius)`; the parallel *work* is the same but the *depth*
collapses to `O(iters · radius)` because all `N` voxels in a pass are independent. That independence — every
voxel's update reads only its own displacement plus the (static) images — is the whole reason the GPU wins.

**Arithmetic intensity / access pattern.** Each Demons force reads a small neighbourhood of `F`, `M`, and one
`u` value: low arithmetic intensity, so the kernels are **memory-bandwidth bound**, and coalesced row-major
access matters. The dose warp is a scattered-read gather (the sample coordinate is data-dependent), which is
why interpolation caching (texture memory in production) helps.

## 4. The GPU mapping

**Thread-to-data.** Every kernel is **one thread per voxel**. The 2-D kernels (DIR force, both Gaussians, the
dose warp) use a `16×16` block over a 2-D grid, so thread `(blockIdx, threadIdx)` owns voxel
`x = blockIdx.x·16 + threadIdx.x`, `y = blockIdx.y·16 + threadIdx.y`. The flat kernels (accumulate, DVH) use a
256-thread 1-D block over `i = blockIdx.x·256 + threadIdx.x`.

**Launch configuration.** `16×16 = 256` threads/block is a multiple of the 32-lane warp and gives the scheduler
8 warps to hide global-memory latency; the 2-D shape mirrors the image so indexing is trivial. Grids use ceiling
division and every kernel guards `x ≥ nx || y ≥ ny` so ragged edge tiles are safe.

**Memory hierarchy.** Everything lives in **global memory**; the field and image are re-read from global each
pass (a shared-memory-tiled Gaussian is Exercise 2). The whole DIR solver stays **resident on the device** — no
host↔device traffic inside the iteration loop — which is the point of GPU DIR. `DemonsParams` is passed **by
value** so it lands in constant/parameter memory, broadcast to every thread for free. The bandwidth bottleneck
is the neighbourhood reads in the force and Gaussian passes.

**Ping-pong buffers.** The separable Gaussian must not read a field another thread is overwriting, so
`smooth-x` reads `d_ux → d_ux2` and `smooth-y` reads `d_ux2 → d_ux` (double-buffered stencil, cf. 6.04 / 14.02).
After the Y pass the smoothed field is back in `d_ux/d_uy`, ready for the next force pass.

**The DVH atomic.** The histogram is a reduction — many threads hit the same bin. We `atomicAdd` **unsigned
integers** (one count per voxel), not floats: integer atomics commute, so the histogram is order-independent and
deterministic (see §5).

```
   image grid (nx × ny)                     DVH (32 integer bins)
   ┌───────────────────────┐                ┌─┬─┬─┬─ ... ─┬─┐
   │  16×16 thread blocks   │   dvh_kernel   │ │ │ │       │ │
   │  ┌────┐┌────┐┌────┐    │  ───────────▶  └─┴─┴─┴─ ... ─┴─┘
   │  │thr.││thr.││... │    │  atomicAdd(&hist[bin(D[i])], 1u)
   │  └────┘└────┘└────┘    │        (integer -> deterministic)
   └───────────────────────┘
```

**Which library does what.** None — deliberately. The catalog mentions cuFFT (for the Gaussian) and cuBLAS (for
B-spline coefficients). This teaching version hand-writes a **separable stencil** Gaussian (clearer, and for our
small `radius` faster than an FFT round-trip) and uses classic **Demons**, not B-splines, so there is no dense
linear solve. §7 explains where the library route pays off. To *hand-roll* a cuFFT Gaussian you would R2C-FFT
the field, multiply by the Gaussian's transfer function, and inverse-FFT — worthwhile only when the kernel is
large (big σ) so `O(N log N)` beats `O(N·radius)`.

## 5. Numerical considerations

- **Precision: FP64 throughout.** DIR is a long iterative solver; double precision keeps the force and the
  accumulated Gaussian stable over 120 iterations and makes the CPU/GPU comparison meaningful. Dose is also
  double (a few Gy with many significant digits).
- **Determinism — the load-bearing design choice.** `demo/run_demo` diffs **stdout**, so stdout must be
  byte-identical every run (PATTERNS.md §3). Two rules:
  1. **Split streams:** deterministic results (from the serial CPU field / integer DVH) → stdout; timings and
     the exact GPU error → stderr.
  2. **Integer histogram.** A *float* `atomicAdd` histogram would sum in nondeterministic thread order and
     drift run-to-run. We add **integers** (a count per voxel), which commute, so the DVH is bit-reproducible
     *and* exactly equals the CPU's. The dose *warp* itself has no cross-thread reduction (each thread writes
     its own output), so it is deterministic too.
- **Race conditions.** None in the per-voxel kernels: each thread writes a distinct output element (force, warp,
  accumulate) — no atomics needed there. Only the DVH is a reduction, handled by integer atomics.
- **FP drift, CPU vs GPU.** The GPU fuses multiply-adds (FMA) differently from the host compiler, so over 120
  iterations the two DVFs can differ by ~1e-5 in principle. In practice on this smooth case they agree to ~1e-15
  (see the demo's `worst DVF diff`), well inside our honest 1e-3 px tolerance.

## 6. How we verify correctness

The trusted baseline is `src/reference_cpu.cpp`: the identical pipeline in plain serial loops. Because every
per-voxel formula lives in `demons.h` / `dose.h` as `__host__ __device__` inlines, the CPU and GPU run
**byte-for-byte-identical math** (PATTERNS.md §2), so agreement is exact-to-rounding, not approximate. `main.cu`
runs both and asserts **three independent checks**:

| Check | Quantity | Tolerance | Why |
|---|---|---|---|
| DVF | max per-component `|u_cpu − u_gpu|` | `≤ 1e-3` px | long iterative solver; FMA drift possible, but result is far below one voxel (PATTERNS.md §4). |
| Dose | max `|D_cpu − D_gpu|` | `≤ 1e-9` Gy | the warp+accumulate is a short computation; matches to near machine precision. |
| DVH | integer counts | **exact (`== 0`)** | integer atomics are order-independent; the histograms must be identical. |

A second, *scientific* check beyond CPU==GPU: the recovered **mean displacement (~4.5 px)** matches the
shift+stretch baked into the synthetic daily image, and the DVH falloff traces the known dose cloud — so we are
validating the physics, not just that two implementations agree. Edge cases exercised: clamp-to-edge sampling at
borders, ragged final tiles (guarded), and the `ε` floor in flat regions.

## 7. Where this sits in the real world

Production ART tools differ from this teaching version in scale and rigor:

- **3-D, DICOM-RT, multimodality.** **Plastimatch** and **CERR** register full 256³–512³ volumes, read/write
  DICOM-RT dose and structures, and handle CBCT↔CT intensity mismatch (Demons-with-SSD assumes matched
  intensities; real DIR uses mutual information or preprocessing).
- **Diffeomorphic / B-spline transforms.** Classic Demons (ours) can fold the field. **Diffeomorphic Demons**
  and **B-spline FFD** guarantee an invertible, smooth transform — B-splines solve for control-point
  coefficients (a linear system where cuBLAS/cuSOLVER earn their keep). **VoxelMorph** learns the DVF with a CNN
  for sub-second daily DIR.
- **Energy/mass-transfer accumulation.** Interpolating dose (our method) does not conserve energy when tissue
  compresses or expands. The **energy/mass-transfer** method deposits each source voxel's energy·mass into its
  deformed neighbours (a *push* with `atomicAdd`), conserving total energy — the physically rigorous choice for
  large deformations, and what AAPM TG-132 discusses. It is Exercise 3.
- **Uncertainty & re-optimization.** DIR error propagates to dose error; production research runs **ensemble
  DIR** for a probabilistic accumulated dose, and tools like **pyRadPlan** re-optimize the plan on the adapted
  anatomy — all under the online-ART **<5-minute** budget that makes the GPU mandatory.

---

## References

- J.-P. Thirion, *Image matching as a diffusion process: an analogy with Maxwell's demons*, Medical Image
  Analysis 2(3), 1998 — the Demons force this project implements.
- T. Vercauteren et al., *Diffeomorphic Demons*, NeuroImage 45(1), 2009 — the invertible variant (§7).
- M. Balik et al. / AAPM **TG-132**, *Use of Image Registration and Fusion Algorithms and Techniques in
  Radiotherapy*, Med. Phys. 2017 — the clinical QA standard for DIR + deformable dose accumulation.
- **Plastimatch** (https://plastimatch.org/) — GPU B-spline DIR + dose warping; the closest production analogue
  to this pipeline.
- **CERR** (https://github.com/cerr/CERR) — a full deformable dose-accumulation pipeline to compare against.
- **VoxelMorph** (https://github.com/voxelmorph/voxelmorph) — learned DIR for fast daily registration.
- **pyRadPlan** (https://github.com/e0404/pyRadPlan) — adaptive plan re-optimization on the adapted anatomy.
