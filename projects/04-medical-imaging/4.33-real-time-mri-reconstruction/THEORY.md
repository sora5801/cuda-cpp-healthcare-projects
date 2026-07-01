# THEORY — 4.33 Real-Time MRI Reconstruction

> The deep dive. Read [README.md](README.md) first for the overview, then this for the
> science → math → algorithm → GPU-mapping → numerics → verification → real-world chain.
> Code references point at [`src/grid_core.h`](src/grid_core.h),
> [`src/reference_cpu.cpp`](src/reference_cpu.cpp), and [`src/kernels.cu`](src/kernels.cu).

---

## 1. The science

An MRI scanner never measures an image directly. As the magnetization precesses, the
receive coil integrates it against a spatial phase set by time-varying gradients, so what
the scanner records is **k-space**: samples of the image's 2D (or 3D) spatial Fourier
transform. A full Cartesian scan visits every k-space line, one per excitation — accurate
but **slow** (seconds to minutes). That is fine for a still knee; it is useless for a
**beating heart** or an **interventional catheter**, where the anatomy changes faster than
a full scan completes.

Real-time MRI trades a full scan for a **continuous, non-Cartesian** acquisition. The
dominant real-time trajectory is **radial**: each excitation reads a straight "spoke"
through the center of k-space at some angle. Two properties make radial ideal for
real-time:

1. **Every spoke samples the center**, where most MR signal energy lives, so even a few
   spokes give a usable (if streaky) image — the acquisition **degrades gracefully**.
2. **Golden-angle ordering** (consecutive spokes 111.25° apart) means *any* contiguous run
   of spokes covers k-space near-uniformly. So we can slide a **window** along the stream
   of spokes and reconstruct a fresh image at every time step, choosing the
   temporal-resolution ↔ image-quality trade-off *after* the scan by changing the window
   size. This is the essence of real-time / **XD-GRASP**-style reconstruction.

The clinical payoff: cardiac function movies, free-breathing scans, and catheter guidance,
all at interactive frame rates — but only if each frame reconstructs in well under 100 ms,
which forces the compute onto a GPU.

## 2. The math

**Forward model.** For a single coil and slice, the (continuous) signal at k-space point
`k = (k_x, k_y)` is the Fourier transform of the image `m(x)`:

```
s(k) = ∫ m(x) · exp(-2πi k·x) dx.
```

Radial sampling evaluates `s(k)` along spokes: spoke `p` at angle `θ_p` reads samples
`k = r · (cos θ_p, sin θ_p)` for readout offsets `r ∈ [-k_max, k_max]`. With the golden
angle, `θ_p = p · 111.25°` (see `golden_angle_rad` in `grid_core.h`).

**Inverse problem: gridding NUFFT.** We want `m(x)` back from non-uniform samples `s(k_j)`.
We cannot inverse-FFT directly because the `k_j` are off-grid. Gridding (Jackson 1991)
approximates the non-uniform inverse transform with four exact, cheap steps:

1. **Density compensation.** Radial samples are dense near the center and sparse at the
   edge; each sample must be weighted by the k-space "area" it represents. For a radial
   trajectory the analytic weight is the **ramp** `w(k) ∝ |k|` — the *same* filter that
   appears in CT filtered backprojection (project 4.01!). See `radial_dcf` in
   `grid_core.h`.

2. **Convolution gridding.** Convolve the density-compensated samples with a small
   interpolation kernel `C` and sample the result on the Cartesian grid:
   `M_grid[g] = Σ_j w(k_j) s(k_j) · C(g - k_j)`. The near-optimal finite-support kernel is
   the **Kaiser-Bessel (KB)** window
   ```
   C(u) = I₀( β · sqrt(1 - (2u/W)²) ) / I₀(β),   |u| ≤ W/2,
   ```
   where `I₀` is the modified Bessel function (`bessel_i0`), `W` is the kernel width in
   cells, and `β` its shape parameter (`kb_beta_for_width`, Beatty 2005). See `kb_weight`.

3. **Inverse FFT.** `m̃(x) = F⁻¹{ M_grid }`. This is the expensive step, done by **cuFFT**
   on the GPU and by a hand radix-2 FFT on the CPU.

4. **Deapodization.** Convolving in k-space **multiplied** the image by `Ĉ(x)`, the
   Fourier transform of the KB kernel. We divide it back out. The KB window's FT has a
   closed form (a sinc-like function):
   ```
   Ĉ(x) ∝ sin( sqrt( (πWx/N)² - β² ) ) / sqrt( (πWx/N)² - β² )
   ```
   (with the `sinh` branch when the argument is negative). See `kb_deapod_1d`. Because the
   KB kernel is separable, the 2D deapodization is the outer product `Ĉ(row)·Ĉ(col)`.

**FFT-shift bookkeeping.** Our gridded k-space has DC at the grid center `n/2`, but a plain
inverse FFT expects DC at index 0 and produces an image with its origin at index 0. The
standard fix brackets the transform with two circular rolls of `n/2` (for even `n`,
`ifftshift == fftshift`):
```
image = FFTSHIFT( IFFT2( IFFTSHIFT( M_grid ) ) ).
```
`IFFTSHIFT` moves DC to index 0 so the FFT reads the right frequency bins; `FFTSHIFT`
re-centers the resulting image. (Getting this wrong shifts the anatomy by `n/2` — a
classic bug we hit and document while building this project.)

**Sliding window.** Frame `f` reconstructs from spokes `[f·stride, f·stride + win)`; each
frame is one full gridding NUFFT. Advancing `stride < win` overlaps windows for temporal
smoothing.

## 3. The algorithm (and complexity)

For each of `n_frames` frames:

```
zero the grid
for each of (win · n_ro) samples in the window:          # the SCATTER
    (kx, ky) = sample_kpos(spoke, readout)               # golden-angle geometry
    v        = sample · radial_dcf(|k|)                  # density compensation
    for each of ~(W+1)² nearby grid cells g:             # KB convolution
        grid[g] += v · kb_weight(|g.x-kx|) · kb_weight(|g.y-ky|)
ifftshift(grid); img = ifft2(grid); fftshift(img)        # transform (cuFFT)
for each pixel: img /= kb_deapod(row)·kb_deapod(col); out = |img|   # deapodize
```

**Complexity per frame.** Gridding scatter: `O(win · n_ro · W²)`. Inverse FFT:
`O(n² log n)`. Deapodize + magnitude: `O(n²)`. For the committed sample
(`win=21, n_ro=64, W=4, n=32`): scatter ≈ `21·64·25 ≈ 34k` weighted adds, FFT ≈
`32²·log 32 ≈ 5k` — both tiny, so the demo is instant. At clinical sizes
(`n≈256`, hundreds of spokes, 32 coils, 3D) the constants explode and the GPU wins big.

**Serial vs. parallel.** The scatter is a sum of independent per-sample contributions, so
it parallelizes perfectly *if* the accumulation into shared grid cells is handled safely
(see §5). The FFT is `O(N log N)` either way but cuFFT extracts far more parallelism than a
serial radix-2 loop.

## 4. The GPU mapping

| Stage | Kernel (`kernels.cu`) | Thread → data | Memory |
|---|---|---|---|
| Gridding scatter | `grid_scatter_kernel` | one thread per (spoke,readout) sample in the window | reads `d_samples` (global), atomic-adds into fixed-point grid (global) |
| Fold + ifftshift | `fold_and_ifftshift_kernel` | one thread per grid cell | reads fixed-point grid, writes complex grid (rolled) |
| Inverse FFT | **cuFFT** `cufftExecC2C(..., CUFFT_INVERSE)` | library-managed | in-place on the complex grid |
| Normalize | `scale_kernel` | one thread per pixel | applies `1/(n·n)` |
| Deapodize + fftshift + magnitude | `deapod_magnitude_kernel` | one thread per output pixel | reads rolled source, writes the frame |

- **Block size 256** — a multiple of the 32-lane warp with enough warps to hide latency on
  sm_75–sm_89.
- **The scatter is the teaching centerpiece.** Sample index `g` maps to
  `spoke = spoke0 + g/n_ro`, `readout = g%n_ro`. The thread computes the sample's grid
  position and density weight, then loops over the `(W+1)²` footprint, atomically adding its
  KB-weighted contribution. Neighboring samples' footprints **overlap**, so many threads hit
  the same cell — hence atomics (§5).
- **cuFFT, not a black box.** `cufftPlan2d(&plan, n, n, CUFFT_C2C)` plans one `n×n`
  complex-to-complex 2D FFT (row-major, stride `n`), exactly our layout. `CUFFT_INVERSE`
  computes `Σ X[k] exp(+2πi k·m / n)` — the same double sum `ifft2_cpu` does by hand — and,
  like ours, leaves it **un-normalized**, so we apply `1/(n·n)` ourselves. Hand-rolling it
  would mean writing and tuning a bit-reversal + butterfly FFT across both axes on the GPU;
  cuFFT does it faster and correctly. We reuse **one plan** for every frame.
- **Memory hierarchy.** All buffers are in global memory; the grid is small enough that
  atomics stay in L2. `Cplx` is layout-identical to `cufftComplex` (asserted with
  `static_assert`), so we `reinterpret_cast` for the FFT with zero copying. A larger project
  would tile the scatter into shared memory per grid tile to cut atomic traffic.
- **Streaming (the real point of "real-time").** Production systems overlap each frame's
  reconstruction with the *acquisition* of the next spokes using **CUDA streams** and
  double-buffering. This teaching version runs the stages sequentially and synchronously so
  every step is observable; Exercise 5 wires up the streamed version.

## 5. Numerical considerations

- **Precision.** FP32, matching real scanners and cuFFT. The KB weight and its
  deapodization FT are evaluated in `double` internally (via `bessel_i0`, `sin`, `sinh`) for
  accuracy, then stored as `float`.

- **Determinism via fixed-point atomics.** Floating-point `atomicAdd` is **not
  associative**: the sum of a grid cell depends on the (nondeterministic) order threads
  arrive, so a `float` grid would differ run-to-run *and* differ from the ordered CPU loop.
  We instead accumulate in **64-bit fixed-point integers** (`to_fixed`/`from_fixed`, scale
  `2²⁰`): quantize each contribution to an integer, sum with integer `atomicAdd`, convert
  back once at the end. Integer addition commutes, so the scatter is **order-independent →
  deterministic and bit-identical to the CPU** (PATTERNS.md §3). CUDA has no signed-64-bit
  `atomicAdd`, so we reinterpret the accumulator as `unsigned long long`; two's-complement
  addition is bit-identical whether read as signed or unsigned.

- **Shared math header.** Every per-sample/per-pixel formula lives once in
  [`grid_core.h`](src/grid_core.h) as `__host__ __device__` inline functions, included by
  both the CPU reference and the GPU kernels. So the two paths run **byte-for-byte identical
  arithmetic**; the only remaining difference is our radix-2 FFT vs cuFFT.

- **Deapodization guard.** `Ĉ(x)` dips toward zero at the image edges; dividing by it would
  amplify edge noise, so `kb_deapod_1d` clamps its magnitude away from zero (a standard
  gridding safeguard).

## 6. How we verify correctness

Two independent checks, both in `main.cu`:

1. **GPU == CPU.** We reconstruct the whole movie on both paths and require the RMS
   difference to be within `TOL_ABS + TOL_REL·peak` (≈`1e-4·peak`). In practice the gridding
   is *exact* (fixed-point integers) so the measured difference is ~`1e-11`, driven only by
   cuFFT-vs-radix-2-FFT rounding over one inverse FFT (PATTERNS.md §4, the "single FFT"
   case). We verify to a physically-negligible tolerance and **say so** rather than claiming
   bit-identity.

2. **The science: does the recon recover the anatomy?** We correlate the last frame with the
   known synthetic phantom (`normalized_correlation`) and require > 0.85. The committed
   sample reaches ≈ 0.96 — strong structural agreement from only 21 sparse spokes. The
   threshold leaves honest margin for the radial **streaking** and the **time-mixing** (the
   window spans spokes acquired at slightly different phantom states) this reduced setup has.

A third, qualitative check is baked into the demo: the per-frame **peak location drifts**
(row 14 → 13 → 12) as the phantom bobs, confirming the sliding window produces a genuine
*dynamic* movie, not a static image repeated.

**Edge cases** handled: ragged last thread block (guarded), samples whose KB footprint
falls off the grid (clipped), the DC sample's density weight (floored so it isn't discarded),
and a non-power-of-two `n` (rejected at load, since the radix-2 FFT needs it).

## 7. Where this sits in the real world

This is a **reduced-scope teaching version** (CLAUDE.md §13). Production real-time MRI
differs in every dimension:

- **Multi-coil parallel imaging.** Real scanners have 8–32 receive coils; reconstruction
  combines them with **GRAPPA/SENSE** to unalias, not just density compensation. Each coil
  is its own gridding NUFFT (embarrassingly parallel across coils and a natural multi-GPU
  split).
- **Iterative & regularized reconstruction.** Simple gridding streaks with few spokes.
  State-of-the-art methods (**GRASP**, **XD-GRASP**, **L+S**, low-rank + sparse) pose
  reconstruction as an *iterative* optimization with a compressed-sensing prior along the
  temporal/cardiac dimension — each iteration runs a NUFFT + adjoint NUFFT like the one here.
  (Sibling project **4.03** builds the Cartesian compressed-sensing solver; combining it with
  this NUFFT is the path to GRASP.)
- **Grid oversampling & better DCF.** Real gridders use 1.5–2× grid oversampling and an
  *iterative* density-compensation (Pipe–Menon) to suppress aliasing (Exercises 1–2).
- **3D + cardiac phase.** Clinical real-time MRI is 4D (3D volume + cardiac phase); the FFT
  and gridding extend to 3D and cuFFT plans a 3D transform.
- **True streaming.** Frameworks like **Gadgetron** run the pipeline acquisition-
  synchronously with CUDA streams so images appear *during* the scan; **BART** provides the
  reference offline NUFFT/GRASP implementations. Learned reconstruction (**MoDL**, unrolled
  networks, cuDNN inference) is the current research frontier.

What transfers exactly from this project: the **gridding NUFFT skeleton** (density comp →
KB grid → FFT → deapodize), the **scatter-with-atomics** GPU pattern, the **deterministic
fixed-point** trick, and the **sliding-window** structure — the load-bearing ideas behind
every real-time MRI reconstructor.
