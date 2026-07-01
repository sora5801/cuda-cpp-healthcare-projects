# THEORY — 4.2 Iterative / Model-Based CT Reconstruction

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See
> [README.md](README.md) for the quick tour and build steps.
>
> _Educational only — not for clinical use._

This project reconstructs a CT image from its X-ray projections **iteratively**:
it repeatedly *simulates* a scan of its current guess, compares to the real
measurement, and corrects the guess. That is the opposite philosophy to
Project 4.01 (Filtered BackProjection), which inverts the data in a single
analytic pass. Iteration costs 20–200× more compute — which is exactly why it
needs a GPU — but it lets us fold in **noise statistics** and a **prior** (here,
total variation), and that is what lets model-based reconstruction produce a
cleaner image at lower radiation dose.

---

## 1. The science

A CT scanner rotates an X-ray source/detector pair around the patient. At each
angle it measures how much the beam is attenuated along thousands of parallel
rays. Beer–Lambert says a ray of initial intensity `I₀` emerges with intensity

```
I = I₀ · exp( −∫ μ(x,y) dℓ )
```

so the **log-attenuation** `−ln(I/I₀) = ∫ μ dℓ` is a *line integral* of the
tissue's linear attenuation coefficient `μ` along the ray. Stacking all those
line integrals (one per angle, per detector bin) gives the **sinogram** `b`. The
reconstruction problem is to recover the 2-D map `μ(x,y)` — the image a
radiologist reads — from `b`.

Why not just filter-and-backproject (FBP, Project 4.01)? Because at **low dose**
(few photons) the sinogram is noisy, and FBP's ramp filter *amplifies* that noise
into streaks. Fewer photons means fewer X-rays through the patient — less
radiation — so a reconstruction that tolerates noise is directly a reconstruction
that lets you scan at lower dose. Iterative model-based reconstruction (MBIR) does
that by (a) trusting well-measured rays more than noisy ones and (b) imposing a
prior belief that the image is *piecewise smooth* (tissue is mostly uniform with
sharp boundaries). Published MBIR reduces noise 30–50% at matched dose vs. FBP.

## 2. The math

Discretize the image into `N²` pixels `x ∈ ℝ^{N²}` and the scan into `M` rays.
The scanner is a **linear operator** `A ∈ ℝ^{M×N²}` (the *system matrix*):
`A[i,p]` is how much pixel `p` contributes to ray `i` (its intersection
length/weight). Then a noiseless scan is `b = A x`. Reconstruction is the inverse
problem: given noisy `b`, find `x`.

We pose it as **penalized weighted least squares (PWLS)**:

```
minimize_x   ½ (Ax − b)ᵀ W (Ax − b)   +   β · R(x)
             └─────── data fidelity ───────┘   └ prior ┘
```

- `W` is a diagonal **statistical weight** (larger for well-measured rays; from
  the Poisson photon count). Setting `W = I` recovers ordinary least squares.
- `R(x)` is the **regularizer / prior**. We use isotropic **total variation**
  `R(x) = Σ ‖∇x‖`, whose gradient smooths flat regions but not edges.
- `β ≥ 0` trades data-fit against smoothness.

**SIRT** (Simultaneous Iterative Reconstruction Technique) is a preconditioned
gradient descent on the data-fidelity term. Its update is

```
x^{k+1} = x^k + λ · C · Aᵀ · R_w · ( b − A x^k )
```

where `R_w = diag(1 / row-sums of A)` and `C = diag(1 / column-sums of A)` are the
SIRT normalization diagonals (each residual bin and each pixel is divided by how
many things touch it), and `0 < λ ≤ 2` is a relaxation/step size. The two matrix
products are the only heavy operations:

- `A x` — **forward projection** (simulate the scan of the current image),
- `Aᵀ r` — **backprojection** (smear the residual back into image space).

`Aᵀ` must be the exact **transpose** (adjoint) of `A`, or the iteration is not a
valid descent and may not converge. After each SIRT update we (i) clamp `x ≥ 0`
(attenuation cannot be negative) and (ii) take one explicit **TV-descent** step,
which is the `β·R(x)` prior applied as a small edge-preserving smoothing.

Symbols: `x` image (pixels, unitless attenuation here), `b` sinogram (line-integral
units), `A` system matrix (weights ∈ [0,1] from linear interpolation), `λ` step
(dimensionless), `N` image side (px), `M = n_angles·n_det` rays.

## 3. The algorithm

```
x⁰ = 0
repeat  iters  times:
    sim   = A x                 # forward project      O(M · pixels_per_ray)
    resid = R_w ⊙ (b − sim)     # weighted residual    O(M)
    grad  = Aᵀ resid            # backproject          O(N² · angles)
    x     = max(0, x + λ · C ⊙ grad)                  # SIRT step + non-negativity
    x     = TV_step(x)          # edge-preserving prior O(N²)
```

**Complexity.** One iteration is dominated by the two projections. In our
voxel-driven implementation each is `O(n_angles · N²)`, so the whole run is
`O(iters · n_angles · N²)`. For the committed sample (48 angles, 48² image, 60
iters) that is ~8M ray–pixel evaluations *per projection direction*; for a
clinical `512³` volume × 1000 views × 100 iters it is ~10¹³ — hopeless serially,
routine on a GPU. Arithmetic intensity is low (a few flops per global-memory
load), so the projections are **bandwidth-bound**, the classic CT profile.

**Serial vs. parallel.** The serial cost above has depth `O(iters)` (each
iteration depends on the previous image). *Within* an iteration, though, every
ray of the forward projection and every pixel of the backprojection is
independent — that is the parallelism the GPU harvests.

## 4. The GPU mapping

We keep the whole reconstruction resident on the device: upload `b`, the trig
tables, and the SIRT weights **once**, then re-launch four kernels per iteration
(the outer loop is a plain C++ `for` on the host — PATTERNS.md §7). Two
thread-to-data mappings appear:

**Forward projection `A` — one thread per detector bin (ray).**
Thread `ray = k·n_det + j` owns `sino[k,j]`. It loops over every pixel (in the
same `py`-outer/`px`-inner order the CPU uses), and for pixels whose ray at angle
`k` lands on bin `j` it adds their linearly-interpolated contribution into a
**private register** `acc`, writing one output at the end. One writer per output
⇒ **no atomics, deterministic**, and — because it reuses the same interpolation
stencil as backprojection — it is the *exact transpose* of `Aᵀ`.

**Backprojection + SIRT update — one thread per pixel (2-D grid).**
Thread `(px,py)` gathers the residual sampled where its ray hits the detector at
each angle (a per-pixel *gather*, identical in spirit to Project 4.01), then
applies `x[p] = max(0, x[p] + λ·C[p]·acc)` in place (safe: each thread touches
only its own pixel).

**TV step — one thread per pixel, ping-pong buffers.**
Each thread reads its 4 neighbours, so an in-place write would race with a
neighbour. We read from `img_in` and write `img_out`, then swap the pointers — the
classic double-buffer used by the stencil flagships 6.04 / 14.02.

```
   sinogram b (M rays)          image x (N×N pixels)
   ┌───────────────┐            ┌───────────────┐
   │ forward: 1 thread          │ backproject+update: 1 thread
   │ per (angle,bin) ─ A ─►      │ per pixel ─ Aᵀ ─► + λC·grad
   │ 1-D grid, 256/block         │ 2-D grid, 16×16 tiles
   └───────────────┘            └──────┬────────┘
            ▲                          │ TV step (ping-pong img↔img2)
            └────────── loop `iters` ──┘
```

**Launch config.** 1-D kernels use 256 threads/block (8 warps, hides latency);
the image kernels use 16×16 = 256-thread tiles that map squarely onto the `N×N`
grid. Both give good occupancy on sm_75…sm_89. **Memory:** inputs live in global
memory; each thread's accumulator is a register; the trig tables (`cosv/sinv`) are
small and read repeatedly — a natural fit for constant/`__ldg` caching (a stated
exercise). No shared memory is needed at this teaching size; a production
ray-driven cone-beam projector tiles the detector into shared memory (the "shared-
memory tile reuse" the catalog notes).

**Libraries.** This teaching version writes the projectors by hand so nothing is a
black box. Production stacks build `A` as a **sparse matrix** and use cuSPARSE for
`A x`/`Aᵀ y`, or use cuFFT for FBP-style filtering inside a hybrid solver; see §7.

## 5. Numerical considerations

- **Precision.** Everything is FP32, matching real CT pipelines (image and
  detector values fit comfortably in single precision, and FP32 doubles GPU
  throughput and halves memory traffic vs. FP64). SIRT is self-correcting — each
  iteration re-measures the residual — so single-precision rounding does not
  accumulate catastrophically.
- **Determinism.** Every kernel writes each output from **one** thread that sums
  in a **fixed order**; there are **no floating-point atomics** anywhere. So the
  GPU result — and the program's stdout — is byte-identical on every run
  (PATTERNS.md §3). We deliberately chose the ray-per-thread forward projector
  (each ray sums its own pixels) precisely to avoid the nondeterministic
  `atomicAdd`-into-shared-bins that a naive voxel-scatter would need.
- **Adjointness.** Forward and backprojection share `interp_stencil`, so `Aᵀ` is
  the true transpose of `A`. If they diverged, SIRT would descend the wrong
  functional and could stall or oscillate.
- **TV stability.** The TV step uses an explicit, tiny step `weight` and an
  `ε`-regularized gradient magnitude `1/√(ε²+‖∇x‖²)` to avoid divide-by-zero on
  flat regions; too large a `weight` oversmooths (a good exercise to observe).

## 6. How we verify correctness

Two independent checks (PATTERNS.md §4):

1. **CPU ⟷ GPU agreement.** `reconstruct_sirt_cpu` runs the identical algorithm
   serially and shares the *exact same per-ray math* via `ct_geometry.h`, so the
   two reconstructions should match. We verify `max|x_GPU − x_CPU| ≤ 2·10⁻³`.
   Why not exact? Over 60 iterations of float projections, the GPU's fused
   multiply-add (FMA) contracts `a*b+c` differently from the host compiler, so the
   images drift by ~`7·10⁻⁴` even though each step is "the same formula." The
   observed error on the sample is `6.8·10⁻⁴`, safely inside a physically
   negligible tolerance — we verify to it and *say so* rather than pretending the
   images are bit-identical (this FMA drift is itself a lesson).

2. **Reconstruction vs. ground truth (science check).** The synthetic sample
   ships the true phantom, so the demo also prints `RMSE(x, truth) ≈ 0.104`. The
   center pixel recovers ≈ 1.0 (the body density) and the central-row profile
   traces the discs — evidence the *math is right*, not just that CPU==GPU.

Edge cases handled: rays that miss the detector contribute nothing (`interp_stencil`
returns false); the ragged last thread block is guarded; unused rays/pixels get a
`0` SIRT weight so they stay inert.

## 7. Where this sits in the real world

This is a deliberately **reduced-scope teaching version** (2-D parallel-beam, a
matrix-free voxel-driven projector, SIRT + a simple explicit-TV step). Production
model-based reconstruction differs in scale and sophistication:

- **Geometry.** Real scanners are 3-D cone-beam (helical); the projector becomes
  the Feldkamp/distance-driven or separable-footprint kernel, and the "shared-
  memory tile reuse for cone-beam geometry" the catalog notes matters a lot.
- **Statistics.** True MBIR uses the **Poisson** log-likelihood (weight `W` from
  photon counts), not plain least squares.
- **Priors & solvers.** Beyond TV: `q`-GGMRF/dictionary/wavelet priors, and
  **ADMM** or **Chambolle–Pock** primal-dual splitting that decouple data-fit and
  prior into GPU-friendly sub-problems; **OS-SART/OS-EM** accelerate by updating
  from ordered subsets of views; **plug-and-play ADMM** swaps the prior for a
  learned denoiser (e.g. DnCNN).
- **Toolkits.** **ASTRA** (GPU projection primitives), **TIGRE** (SART/CGLS/OS-TV
  on GPU), **ODL** (variational framework over ASTRA), and **LLNL LEAP**
  (penalized-likelihood on GPU) implement the full versions — study them to see
  how the ideas here scale.

---

## References

- ASTRA Toolbox — <https://github.com/astra-toolbox/astra-toolbox> — GPU forward/
  backprojection primitives; read to see a production matrix-free projector API.
- TIGRE — <https://github.com/CERN/TIGRE> — SART, CGLS, OS-TV with CUDA; a clean
  reference for ordered-subset acceleration and TV regularization on GPU.
- ODL (Operator Discretization Library) — <https://github.com/odlgroup/odl> —
  how a variational problem (`½‖Ax−b‖²+βR(x)`) is assembled from operators.
- LLNL LEAP — <https://github.com/LLNL/LEAP> — GPU penalized-likelihood CT; the
  statistical/Poisson weighting this teaching version omits.
- A. Kak & M. Slaney, *Principles of Computerized Tomographic Imaging* — the
  standard text for the Radon transform, FBP, and the algebraic (ART/SIRT) methods.
- J. A. Fessler, *Model-Based Image Reconstruction for MRI/CT* (review) — the PWLS
  framework, statistical weights, and edge-preserving priors in depth.
