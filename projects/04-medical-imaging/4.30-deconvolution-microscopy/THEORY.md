# THEORY — 4.30 Deconvolution Microscopy

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## The science

A fluorescence microscope is an imaging system, and like every optical system it
is **band-limited**: it cannot resolve detail finer than roughly half the
wavelength of light (the diffraction limit, ~200–300 nm for visible light). The
practical consequence is that a single, infinitely small point of light in the
specimen does **not** map to a single pixel in the image — it spreads into a small,
fuzzy blob. That blob is the **Point Spread Function (PSF)**: the image the
microscope produces of an ideal point source.

Because the microscope is (to good approximation) **linear and shift-invariant**
— every point in the field is blurred by the *same* PSF, and the blurs add up —
the entire recorded image is the true specimen **convolved** with the PSF:

```
   observed  =  truth  ⊛  PSF   (+ noise)
```

So the recorded image is a blurred version of reality. **Deconvolution** is the
inverse problem: given the observed (blurry) image and a model of the PSF, recover
a sharper estimate of `truth`. Done well, it visibly improves resolution and
contrast — which is why it ships in commercial microscopes (Zeiss, Leica) and
open-source tools (DeconvolutionLab2, Huygens, CSBDeep).

The catch: the noise in fluorescence imaging is **Poisson** (you are *counting
photons*; few photons → shot noise), and naive inversion (dividing spectra) blows
that noise up catastrophically. We need an inversion that respects the noise model.
That is **Richardson-Lucy**.

---

## The math

### The forward model

Let `f` be the (unknown) true image, `h` the PSF (`Σ h = 1`, intensity-conserving),
and `g` the observed image. Discretized and treating convolution as **circular**
(periodic) for now:

```
   g[x]  =  (h ⊛ f)[x]  =  Σ_x'  h[x - x'] · f[x']
```

Each observed pixel `g[x]` is the *expected* photon count; the actual count is a
Poisson random variable with that mean.

### Richardson-Lucy as maximum likelihood

We want the `f ≥ 0` that is **most likely** to have produced the observed photon
counts under Poisson statistics. The Poisson log-likelihood of `f` given `g` is

```
   L(f)  =  Σ_x [ g[x] · log( (h⊛f)[x] )  −  (h⊛f)[x] ]   (+ const)
```

Maximizing `L` over `f ≥ 0` has no closed form, but the **Expectation-Maximization**
algorithm gives a beautifully simple **multiplicative** fixed-point iteration —
this is Richardson-Lucy (Richardson 1972; Lucy 1974):

```
   f_{k+1}  =  f_k  ·  [  h^T ⊛ ( g / (h ⊛ f_k) )  ]
```

where `h^T` is the **flipped** PSF `h^T[x] = h[-x]` (the adjoint of the convolution
operator `H`). Read it left to right:

1. `h ⊛ f_k` — blur the current estimate (the **forward model**).
2. `g / (h ⊛ f_k)` — the per-pixel **ratio** of observed to predicted. =1 where we
   already match the data, >1 where we under-predict, <1 where we over-predict.
3. `h^T ⊛ (…)` — **back-project** that ratio through the adjoint PSF.
4. `f_k · (…)` — **multiply** the estimate by that correction.

Two properties make RL the workhorse:

- **Non-negativity is automatic.** A product of non-negatives stays non-negative,
  so `f` never goes negative — which a fluorescence intensity must never do.
- **Intensity is conserved.** Because `Σ h = 1` and `h^T` sums to 1 too, the total
  `Σ f` is preserved (up to boundary effects). The demo prints this as a sanity
  check (`observed=8708.00 → deconvolved=8708.00`).

### The convolution theorem (why FFTs)

Both convolutions are the expensive part. The **convolution theorem** says circular
convolution in space equals pointwise multiplication in frequency:

```
   h ⊛ f   =   IFFT(  FFT(h) · FFT(f)  )
   h^T ⊛ r =   IFFT(  conj(FFT(h)) · FFT(r)  )      ← flipping in space = conjugating in frequency
```

So `FFT(h)` (the **transfer function** `H(k)`, the Fourier transform of the PSF) is
computed **once**; each iteration is two FFTs, two pointwise multiplies, two inverse
FFTs. The adjoint costs nothing extra: multiply by `conj(H)` instead of `H`.

---

## The algorithm

```
build PSF h (Gaussian), normalize Σh = 1
H ← FFT(h_embedded)                 # transfer function, computed once
f ← mean(g)  (flat, non-negative initial estimate)
repeat K times:
    blurred    ← IFFT( FFT(f) · H )          # forward model      (conv #1)
    ratio      ← g / blurred                  # per-pixel, guarded
    correction ← IFFT( FFT(ratio) · conj(H) ) # back-projection    (conv #2)
    f          ← f · correction               # per-pixel, clamp ≥ 0
return f
```

**Complexity** (image of `N` pixels, PSF of `K` taps, `K` iterations):

| Step | Direct convolution (CPU ref) | FFT convolution (GPU) |
|---|---|---|
| One convolution | `O(N·K)` | `O(N log N)` |
| Per RL iteration | `O(N·K)` | `O(N log N)` |
| Whole run | `O(K · N · K_taps)` | `O(K · N log N)` |

For a 2,048³ volume with a wide PSF, `K_taps` is huge and direct convolution is
hopeless; FFT convolution is the only practical route — and the FFT is what the GPU
does best.

---

## The GPU mapping

This project is the **"use a CUDA library"** pattern (PATTERNS.md §1, like flagship
`8.03`). The structure:

```
        host (main.cu)                         device
   ┌────────────────────┐
   │ load g, build PSF  │  ── H2D ──▶  d_obs, d_est, d_psf_img
   └────────────────────┘
                         cufftExecD2Z(d_psf_img) ─▶ d_psf_spec  (H(k), once)

   per iteration (all on device):
     cufftExecD2Z(d_est)         ─▶ d_spec          # FFT(estimate)      [cuFFT]
     complex_mul_psf(conj=0)     : d_spec ·= H · 1/N                     [custom kernel]
     cufftExecZ2D(d_spec)        ─▶ d_real          # IFFT → blurred     [cuFFT]
     ratio_kernel                : d_real = g / blurred                  [custom kernel]
     cufftExecD2Z(d_real)        ─▶ d_spec          # FFT(ratio)         [cuFFT]
     complex_mul_psf(conj=1)     : d_spec ·= conj(H) · 1/N               [custom kernel]
     cufftExecZ2D(d_spec)        ─▶ d_real          # IFFT → correction  [cuFFT]
     update_kernel               : d_est ·= correction (clamp ≥0)        [custom kernel]
```

**What cuFFT does (not a black box).** `cufftPlan2d(plan, h, w, CUFFT_D2Z)` builds a
plan for a 2-D **real-to-complex** double-precision FFT of an `h × w` image.
`cufftExecD2Z` computes the standard 2-D DFT

```
   F[ky,kx] = Σ_{y,x} f[y,x] · exp(−2πi (ky·y/h + kx·x/w))
```

Because the input is real, the output is **Hermitian-symmetric**, so cuFFT stores
only the non-redundant half — `h × (w/2+1)` complex bins. That halves the memory and
the pointwise-multiply work. Hand-rolling a competitive batched mixed-radix FFT is
hundreds of lines and a research project in itself; cuFFT gives us a tuned one.

**Memory hierarchy.** The working buffers (`d_obs`, `d_est`, `d_real` real; `d_spec`,
`d_psf_spec` complex) live in **global memory**. The custom kernels are element-wise
streaming passes — each thread reads its own index and writes it back — so they are
**bandwidth-bound**, and the natural "one thread per element" mapping (block = 256,
`grid = ceil(count/256)`) already saturates memory bandwidth. There is no data reuse
to justify shared memory here; the FFT (inside cuFFT) is where the clever
shared-memory tiling happens, and that is cuFFT's job, not ours.

**cuFFT is unnormalized.** A forward+inverse round trip multiplies every value by
`N = w·h`. We fold the `1/N` into `complex_mul_psf` (one multiply we were doing
anyway), so the IFFT output comes out correctly scaled with no extra kernel.

**One stored spectrum, both operators.** We never FFT a flipped PSF. The forward
blur multiplies by `H`; the adjoint back-projection multiplies by `conj(H)` — and
conjugating a complex number is just negating its imaginary part, a branch inside
the same kernel (`conj_psf`). So `H` and `H^T` share one stored transfer function.

---

## Numerical considerations

- **Precision: double throughout.** RL is iterated many times and the multiplicative
  update compounds error. We use double-precision cuFFT (`CUFFT_D2Z`/`CUFFT_Z2D`,
  `cufftDoubleComplex`) so the GPU matches the double-precision CPU reference
  closely. The images are tiny, so double costs nothing meaningful here. (Production
  code often uses single precision for speed/memory — see exercise 5.)
- **Division guard.** `ratio = g / blurred` divides by a quantity that can be ~0 in
  dark background. `rl_ratio()` floors the denominator at `RL_EPS = 1e-12` so a dark
  pixel cannot produce `inf`/`NaN` and poison the whole image through the next FFT.
- **Non-negativity clamp.** In exact arithmetic the update is a product of
  non-negatives, but FFT round-off can yield a tiny negative; `rl_update()` clamps to
  0. A pixel driven to exactly 0 stays 0 forever (multiplicative update) — acceptable
  here, but a reason real code adds a small background floor.
- **Circular vs. linear convolution.** We use **circular** (periodic) convolution
  because that is *exactly* what multiplying FFTs computes. This makes the CPU
  reference (direct circular convolution) and the GPU (FFT) the **same operator**, so
  they are comparable pixel-for-pixel. The price: features wrap around the image
  border. Real deconvolution **zero-pads** the image to at least `N + K − 1` and/or
  apodizes the edges to convert circular into linear convolution and kill wrap-around
  artifacts (an excellent extension).
- **Determinism.** Every kernel is element-wise (no `atomicAdd`, no order-dependent
  reduction), so the GPU result is bit-reproducible run to run. The report rounds
  scalars to a few decimals so the last (FFT-noise) digits never flip the stdout
  bytes (PATTERNS.md §3).

---

## How we verify correctness

Two independent checks:

1. **GPU vs. CPU (the headline gate).** `reference_cpu.cpp` runs the *same* RL
   iteration with a **direct** circular convolution (a transparent quadruple loop),
   and shares the *same* per-pixel `rl_ratio()`/`rl_update()` math via `rl_core.h`
   (the `__host__ __device__` idiom, PATTERNS.md §2). So the GPU and CPU differ
   **only** in *how they convolve* — FFT vs. direct. `main.cu` asserts the worst
   absolute per-pixel error is below `atol = 1e-6`.

   In practice the error is ~`1e-13` — essentially round-off. Why a `1e-6` tolerance
   and not `0`? FFT convolution and direct convolution are the *same map
   mathematically* but a *different sequence of floating-point operations*, and the
   GPU's fused-multiply-add reorders sums; over 30 iterations these diverge by a few
   ULPs (PATTERNS.md §4). `1e-6` on intensities of `O(1..100)` is physically
   negligible and honest.

2. **Science checks (the report).** The demo also prints quantities that validate the
   *algorithm*, not just CPU==GPU agreement:
   - **Sharpness up ~6×** — deconvolution genuinely restored high-frequency detail.
   - **Bright beads recovered** — the diagonal-pixel fingerprint shows the synthetic
     point sources sharpened from blobs back toward points.
   - **Total intensity conserved** (`8708.00 → 8708.00`) — RL preserves the photon
     budget, exactly as the math promises.

---

## Where this sits in the real world

The catalog's full CUDA pattern is *"cuFFT 3-D in-place FFT for PSF convolution;
custom kernel for the R-L multiplicative update; texture memory for the PSF; batched
cuFFT for simultaneous channel deconvolution; pinned memory for streaming large
volumes."* This teaching version implements the **core** (2-D cuFFT + custom RL
kernels); here is how production differs:

- **3-D, huge.** Real deconvolution runs on **z-stacks** (volumes), up to ~2,048³.
  The only change in principle is `cufftPlan3d` and 3-D indexing; the bottleneck
  becomes **memory** — a 2,048³ double-complex spectrum is ~137 GB, so real code
  **tiles** the volume and **streams** tiles through **pinned (page-locked) host
  memory** for fast overlapped H2D/D2H copies.
- **Batched channels.** Multi-channel (color) or multi-tile data uses **batched**
  cuFFT plans (`cufftPlanMany`) to transform many images in one launch — exactly the
  batching flagship `8.03` uses for EEG channels.
- **Measured / blind PSF.** We assume a known Gaussian PSF. Real workflows **measure**
  the PSF from sub-resolution fluorescent **beads**, or do **blind deconvolution**
  (jointly estimating PSF and image — a second optimization variable, more
  iterations). DeconvolutionLab2 and Huygens implement both.
- **Regularization.** Plain RL amplifies noise as iterations grow. Production adds
  **Total-Variation** or Tikhonov regularization, or accelerated RL (vector-extrapolated
  RL), to stop the noise blow-up — see the exercises.
- **Learned restoration.** CSBDeep/CARE replaces the iterative solver with a trained
  CNN: faster at inference and able to learn priors classical RL cannot, at the cost
  of needing training data and careful validation (it can hallucinate structure).
- **Texture memory for the PSF.** For *spatially varying* PSFs (the blur changes
  across the field), the PSF is sampled with hardware-interpolated **texture memory**;
  our shift-invariant PSF needs none.

The lesson to carry away: deconvolution is an **inverse problem** solved by **iterating
a forward model**, and the inner loop is **FFT convolution** — which is why a fast,
trustworthy FFT library (cuFFT) is the whole game.
