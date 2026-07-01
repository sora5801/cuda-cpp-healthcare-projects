# THEORY — 4.29 Light-Sheet Microscopy Reconstruction

> Reduced-scope teaching version: **2D single-view Richardson-Lucy deconvolution**
> with a Gaussian point-spread function, computed in the Fourier domain with
> **cuFFT**. The full multi-view / multi-TB pipeline is described in
> "Where this sits in the real world". Read `README.md` first for the overview.
>
> _Educational only — not for clinical use._

---

## The science

**Light-sheet fluorescence microscopy (LSFM)** — also called **selective plane
illumination microscopy (SPIM)** — images a specimen (a developing embryo, a
cleared brain, an organoid) by shining a *thin sheet of laser light* through one
plane at a time and photographing the fluorescence emitted from just that plane
with a camera looking perpendicular to the sheet. Sweeping the sheet through the
sample builds a 3D stack, plane by plane, with very low photobleaching — which is
why LSFM can film living development for days.

Two physical facts make **reconstruction** necessary before the images are usable:

1. **Blur.** No optical system is perfect. A single point of light (one
   fluorophore) does not image to a single pixel; it spreads into a small blob
   called the **point-spread function (PSF)**. Every recorded image is the true
   fluorophore distribution *convolved* with the PSF. In LSFM the PSF is strongly
   anisotropic (worse along the detection axis), which is why multiple *views*
   (rotations of the sample) are fused — each view is sharp in a different
   direction.

2. **Photon noise.** Fluorescence is faint; the camera counts a small number of
   photons per pixel, so the noise is **Poisson** (shot noise), not Gaussian.

**Deconvolution** undoes the blur — it estimates the true image `x` given the
blurry measurement `b` and the known PSF `h`. The classic, still-dominant method
is **Richardson-Lucy (RL)**, precisely because it is the maximum-likelihood
estimator for *Poisson* noise, and because it is multiplicative and therefore
keeps the recovered intensities non-negative (you cannot have negative photons).

This project implements RL on a single 2D plane with an isotropic Gaussian PSF —
the didactic heart of the LSFM reconstruction pipeline.

---

## The math

### Forward model

The microscope forms the measured image `b` from the true image `x` by
convolution with the PSF `h`, then Poisson sampling:

```
b  =  Poisson( h * x )          ( * denotes 2D convolution )
```

### Richardson-Lucy update

Maximizing the Poisson likelihood of `x` given `b` by expectation-maximization
yields the **multiplicative** iteration

```
                 ⎡          b        ⎤
x_{k+1}  =  x_k · ⎢ h^T  *  ─────────  ⎥
                 ⎣        h * x_k     ⎦
```

where

- `h * x_k` is the current estimate **re-blurred** through the PSF — i.e. what the
  camera *would* see if `x_k` were the truth (the forward model);
- `b / (h * x_k)` is the per-pixel **correction ratio**: >1 where the prediction
  is too dim, <1 where it is too bright, exactly 1 where it matches;
- `h^T *` **back-projects** that ratio through the *flipped* PSF `h^T` (the adjoint
  of convolution, which is correlation);
- the outer `x_k ·` and the `/` are **element-wise**.

**Fixed point.** When `x_k` explains the data, the ratio is 1 everywhere, the
back-projection is 1, and `x_{k+1} = x_k` — RL has converged. Because `h` sums to
1, each step **conserves total flux**: `sum(x_{k+1}) = sum(x_k)` (the demo prints
this as `flux ratio 1.0000`, a live check that the implementation is correct).

### Convolution via the Fourier transform (the GPU lever)

Direct 2D convolution of an `N`-pixel image with an `N`-pixel kernel costs
`O(N^2)`. The **convolution theorem** replaces it with three transforms and one
pointwise multiply:

```
h * x   =  IFFT( FFT(h) · FFT(x) )                 (convolution)
h^T * r =  IFFT( FFT(r) · conj(FFT(h)) )           (correlation = adjoint)
```

The Fourier transform of a *real* image is Hermitian-symmetric, so we only store
half the spectrum (`W/2+1` columns) — the real-to-complex (R2C/D2Z) layout. Each
FFT is `O(N log N)`, so one RL iteration drops from `O(N^2)` to `O(N log N)`. That
is the entire reason LSFM deconvolution is done in k-space.

---

## The algorithm

Notation: image is `H x W`, `N = H*W` pixels. Let `Hf = FFT(psf)` (computed once).

```
x <- flat image at mean(b)                         # unbiased initial estimate
Hf <- FFT(psf)                                      # PSF spectrum, precomputed ONCE
repeat `iters` times:
    reblur  <- IFFT( FFT(x)     · Hf        ) / N   # forward model   (convolution)
    ratio   <- b / max(reblur, eps)                 # per-pixel correction  (rl_ratio)
    correct <- IFFT( FFT(ratio) · conj(Hf)  ) / N   # back-project    (correlation)
    x       <- max(x · correct, 0)                  # multiplicative update (rl_apply)
```

**Complexity.**

| | per convolution | per RL iteration | total |
|---|---|---|---|
| Direct (CPU reference) | `O(N^2)` | `O(N^2)` (2 convs) | `O(iters · N^2)` |
| FFT (GPU, cuFFT)       | `O(N log N)` | `O(N log N)` | `O(iters · N log N)` |

The CPU reference deliberately uses the **direct** `O(N^2)` DFT: slow, but so
simple it is obviously correct — the whole point of a reference. On the tiny
`32 x 32` sample it takes ~100 ms; on a real `2048 x 2048` plane it would take
*hours*, which is why the GPU FFT path exists.

---

## The GPU mapping

**Pattern:** *use a CUDA library for the solved sub-problem* (PATTERNS.md §1, the
`8.03` cuFFT flagship) — here the FFT — and hand-write only the small element-wise
kernels around it.

- **cuFFT** does the forward (`cufftExecD2Z`) and inverse (`cufftExecZ2D`)
  transforms. `D2Z` = real double → complex double; `Z2D` = the inverse. We build
  the two 2D plans once with `cufftPlan2d(H, W, ...)`. cuFFT does **not** normalize,
  so a forward+inverse pair scales the data by `N`; we fold the `1/N` into the
  complex-multiply kernel. *Hand-rolling this* would mean a radix-mixed 2D FFT with
  bit-reversal and twiddle tables — hundreds of lines, far slower than the vendor's
  tuned kernels (CLAUDE.md §6.1.6 — no black boxes, but no reinventing the FFT).
- **`complex_mul_scaled`** (one thread per frequency bin, `nc = H·(W/2+1)` bins):
  multiplies two spectra element-wise, applies the `1/N` scale, and optionally
  conjugates the second operand to switch convolution ↔ correlation (the RL
  adjoint). This is the only "real" arithmetic kernel.
- **`ratio_kernel`** and **`update_kernel`** (one thread per spatial pixel, `N`
  pixels): the two per-pixel RL steps, calling the **shared** `rl_ratio` /
  `rl_apply` from `rl_core.h` — the *same* functions the CPU reference calls, so
  the math is identical on both sides (the HD-macro idiom, PATTERNS.md §2).

**Thread-to-data map.** Every kernel is the fundamental "grid of 1-D threads over
a 1-D array": thread `i = blockIdx.x * blockDim.x + threadIdx.x`, guarded by
`if (i < n)` for the ragged last block. Block size 256 (8 warps) is a solid
occupancy default on `sm_75..sm_89`.

```
   measured b ─┐
               │   (per RL iteration, all on the GPU)
   estimate x ─┼─► cuFFT D2Z ─► [complex_mul_scaled · Hf] ─► cuFFT Z2D ─► reblur
               │                                                            │
               │                              ratio_kernel  b / reblur  ◄───┘
               │                                    │
               │        cuFFT D2Z ◄── ratio ────────┘
               │            │
               │   [complex_mul_scaled · conj(Hf)] ─► cuFFT Z2D ─► correction
               │                                                       │
               └──────────────── update_kernel  x · correction  ◄──────┘
```

**Memory hierarchy.** All buffers live in **global memory**; the access pattern is
perfectly coalesced (thread `i` touches element `i`), so this kernel is
**bandwidth-bound**, not compute-bound — exactly the regime FFT-heavy pipelines run
in. The PSF spectrum `Hf` is computed once and reused across all iterations (a
small but real saving). No shared memory or atomics are needed because every
element is independent.

**Why not a spatial-domain stencil kernel?** For a small PSF a shared-memory tiled
convolution (like `7.10`) can win, but LSFM PSFs are large and multi-view fusion
mixes several PSFs — the FFT approach is `O(N log N)` regardless of PSF size and is
what the production tools use, so it is the right lesson here.

---

## Numerical considerations

- **Precision.** Both paths are **FP64** (double). We deliberately chose cuFFT's
  double-precision `D2Z`/`Z2D` over the single-precision `R2C`/`C2R` so that GPU
  and CPU differ only by *transform round-off*, not by precision. (Real
  deconvolution engines often use FP32 for speed and memory; we note that in the
  README as an exercise.)
- **The `eps` clamp.** RL divides `b` by the re-blurred estimate; where that
  estimate is ~0 the ratio would blow up. Both sides clamp the denominator to the
  *same* `RL_EPS = 1e-7` (in `rl_core.h`) so the clamp fires at identical pixels —
  essential for exact agreement.
- **Non-negativity.** The multiplicative update can drift slightly negative from
  FFT round-off; `rl_apply` clamps at 0 on both sides, keeping the estimate a valid
  image.
- **Determinism.** No atomics and no floating-point reductions in a
  non-deterministic order — every kernel is a pure element-wise map, and the
  summary statistics (`sum`, `L2`) are accumulated left-to-right in fixed order. So
  **stdout is byte-identical every run** (verified), which is what lets the demo
  diff it.
- **Adjoint convention.** The correlation (RL back-projection) is
  `IFFT(FFT(ratio)·conj(FFT(psf)))`. The direct-DFT reference must use the matching
  index formula `sum_{r,c} ratio[r,c]·psf[(r-p) mod H, (c-q) mod W]` **with the
  ratio as the un-conjugated operand**. Getting the argument order right here was
  the one subtle bug worth flagging — see "How we verify correctness".

---

## How we verify correctness

Three independent checks, from weakest to strongest:

1. **GPU vs CPU agreement.** `main.cu` runs the direct-DFT RL (CPU) and the cuFFT
   RL (GPU) and compares three order-independent statistics (`sum`, `max`, `L2`).
   The worst relative difference is **~1.4e-15** — machine precision — because both
   are FP64 and share the PSF and `rl_core.h`. We verify to a documented **1e-9**
   relative floor (PATTERNS.md §4: same math on both sides → tolerance near machine
   epsilon, not a loose physical tolerance).

2. **Flux conservation.** RL with a unit-sum PSF must conserve total intensity. The
   demo prints `flux ratio 1.0000` — `sum(output) == sum(input)` to 4 decimals. If
   a convolution index or the adjoint were wrong, the sum would drift or explode
   (it did, during development, until the adjoint argument order was fixed — a
   genuine, instructive bug).

3. **Recovery of a known ground truth.** The synthetic sample is a known set of
   bright "beads" blurred by the *same* Gaussian. A correct RL run must **sharpen**
   them back up: the demo reports `peak x2.19`, `L2 x1.24` — contrast measurably
   restored. This validates the *science*, not just CPU==GPU numerics.

During development the convolution/correlation conventions were also cross-checked
against a NumPy FFT reference (`IFFT(FFT·FFT)` and `IFFT(FFT·conj(FFT))`) to nail
down the exact index formulas the direct DFT must use.

---

## Where this sits in the real world

This project is a **reduced-scope teaching version** (CLAUDE.md §13). Real LSFM
reconstruction, as done by **BigStitcher** (Fiji/ImageJ) and **DeconvolutionLab2**,
adds several dimensions this single-plane demo omits:

- **3D, not 2D.** The transforms are 3D FFTs over `1000^3`+ voxel sub-volumes;
  cuFFT does 3D plans natively, but memory and streaming dominate.
- **Multi-view fusion.** Several rotated views, each with its own anisotropic PSF,
  are deconvolved *jointly* (the Bayesian multi-view RL of Preibisch et al., 2014),
  so each direction is sharpened by the view that sees it best. Content-weighted
  (entropy-based) fusion blends the views; cuBLAS handles the view-weight matrix
  products.
- **Tile stitching.** Large specimens are acquired as overlapping tiles aligned by
  **phase correlation** (a peak-finding FFT cross-correlation) — another cuFFT use.
- **Blind / measured PSFs.** The PSF is estimated (blind deconvolution) or measured
  from fluorescent beads, not assumed Gaussian.
- **Scale & streaming.** Datasets are terabytes; production code decomposes the
  volume across **z-planes and multiple GPUs**, uses **pinned host memory** to
  stream tiles, and stores data as chunked N5/Zarr. Modern pipelines increasingly
  add learned denoisers (**CARE/CSBDeep**, **Noise2Void**) as a pre- or post-step.

The core RL iteration you see here — re-blur, ratio, back-project, multiply, all in
k-space — is *exactly* the inner loop those tools run. The rest is engineering
around this mathematical kernel.

---

## References

- **Richardson (1972)** & **Lucy (1974)** — the original iterative deconvolution;
  the update this project implements.
- **Preibisch et al., *Nature Methods* (2014)** — efficient Bayesian *multi-view*
  deconvolution; the algorithm behind BigStitcher's fusion. Read for how several
  PSFs combine.
- **Hörl et al., *Nature Methods* (2019)** — **BigStitcher**; the practical
  GPU-accelerated LSFM stitching + fusion pipeline. See the catalog "Prior art".
- **DeconvolutionLab2** (Sage et al.) — a multi-algorithm deconvolution reference
  implementation; good for comparing RL against Tikhonov/Landweber.
- **CSBDeep/CARE** and **Noise2Void** — learned restoration/denoising for LSFM; the
  modern complement to (not replacement for) model-based deconvolution.
- **NVIDIA cuFFT documentation** — the D2Z/Z2D real-transform layout and plan API
  used here.
