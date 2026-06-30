# THEORY — 2.31 Cryo-EM Tilt-Series Alignment & Tomogram Reconstruction

> For a reader who knows C++ but is new to CUDA and to cryo electron tomography.
> See [README.md](README.md) for the tour and build. _Educational only — a
> reduced-scope, 2-D teaching version. Not for any clinical or research-grade use._

## 1. The science

Cryo electron tomography (cryo-ET) images a single, frozen-hydrated biological
specimen — a slab of cell, a virus, a molecular machine *in situ*. The microscope
**tilts** the specimen to a series of angles (typically every few degrees from
roughly −60° to +60°) and records a 2-D projection image at each tilt. That stack
of images is the **tilt series**. From it we reconstruct the specimen's 3-D
density (the **tomogram**).

Three problems stand between the raw tilt series and a clean tomogram, and a real
pipeline solves all three:

1. **Frame / beam-induced motion** — the electron beam makes the thin ice move
   during each exposure (corrected by motion-correction like MotionCor2).
2. **Tilt-series alignment** — between exposures the stage drifts, so projections
   are translated (and slightly rotated/scaled) relative to each other; they must
   be registered to a common frame before reconstruction.
3. **Tomogram reconstruction** — invert the aligned projections into 3-D density,
   classically by **Weighted Back-Projection (WBP)**.

A fourth, unavoidable issue is the **missing wedge**: because the holder cannot
tilt past ~±60°, a wedge of Fourier space is never measured, smearing the
reconstruction along the beam axis. (Deep-learning tools such as IsoNet try to
*fill* the wedge after the fact.)

This project implements a **reduced-scope, 2-D** version of steps 2 and 3:
translational tilt-series alignment by cross-correlation, then WBP of a single
slice with a **cuFFT** ramp filter.

## 2. The math

A projection is a **line integral** of the density `f`. In 2-D, the projection at
tilt angle `θ` and detector position `s` is the **Radon transform**:

```
p(θ, s) = ∫ f(x, y) δ(x·cosθ + y·sinθ − s) dx dy
```

The **Fourier-slice theorem** says the 1-D Fourier transform of `p(θ, ·)` is a
radial slice, at angle `θ`, of the 2-D Fourier transform of `f`. Inverting the
polar→Cartesian change of variables introduces the Jacobian `|ω|`, giving
**weighted back-projection**:

```
f(x, y) = Σ_k  ( p(θ_k, ·) * h )( x·cosθ_k + y·sinθ_k ) · Δθ
```

where `*` is 1-D convolution and `h` is the **ramp filter** with frequency
response `|ω|` (Ram-Lak). "Weighted" = the ramp; "back-projection" = the angular
sum. Without `h`, the sum is `1/r`-blurred.

**Alignment.** If projection `k` is the true projection drifted by `d_k` detector
bins, two adjacent projections (only a few degrees apart) have nearly identical
content, so their **cross-correlation**

```
CC_{k,k-1}(L) = Σ_j  p_{k-1}[j] · p_k[j + L]
```

peaks at `L = d_k − d_{k-1}` — the *relative* drift. Accumulating those relative
lags outward from a reference projection recovers each absolute drift `d_k`.

## 3. The algorithm

```
# Step 1 — tilt-series alignment (sequential cross-correlation)
ref = argmin_k |tilt_k|
shift[ref] = 0
for k from ref-1 down to 0:    shift[k] = shift[k+1] + argmax_L CC(p_{k+1}, p_k)
for k from ref+1 up to K-1:    shift[k] = shift[k-1] + argmax_L CC(p_{k-1}, p_k)
aligned[k] = translate(p_k, shift[k])

# Step 2 — ramp filter (per projection row), in the frequency domain
for each aligned row:  X = FFT(row);  X *= ramp(|f|);  filtered = IFFT(X)

# Step 3 — weighted back-projection (per output pixel)
for each pixel (x,y):
    f(x,y) = (pi/K) * Σ_k  interp( filtered_k, x·cosθ_k + y·sinθ_k )
```

**Complexity.**
- Alignment: `O(K · n_det · W)` for a search window `W` (here a small ±8-bin scan).
- Ramp filter: `O(K · n_det log n_det)` with the FFT.
- Back-projection: `O(img² · K)` — the dominant term, and the part we put on the
  GPU. In 3-D it is `O(vox³ · K)`, where the GPU becomes essential.

## 4. The GPU mapping

Two GPU teaching points, matching the catalog's named pattern ("custom CUDA WBP
kernel … cuFFT for filter application").

**(a) Ramp filter with cuFFT.** We run a **batched** real-to-complex FFT over all
`K` projection rows at once (`cufftPlan1d(…, CUFFT_R2C, K)`), multiply each
spectrum by the ramp weight with a tiny element-wise kernel (one thread per
spectral bin, `i % nf` selects the weight), then a batched complex-to-real inverse
FFT. cuFFT owns the `O(n log n)` transform; we own the *physics* (the `|f|` ramp).
What hand-rolling would cost: a mixed-radix FFT with bit-reversal and twiddle
factors — exactly the solved primitive a library should provide.

**(b) Back-projection as a per-pixel gather.** One thread per output pixel, on a
2-D grid of 16×16 blocks tiling the slice. Thread `(px,py)` owns pixel `(px,py)`:

```
  for each tilt k:
     s    = wx*cos[k] + wy*sin[k]      # where this pixel's ray hits detector k
     fidx = s/ds + center             # fractional detector index
     acc += lerp(filtered_k[fidx])    # linear interpolation in the detector
  slice = acc * (pi / K)
```

**Memory hierarchy.** `filtered`, `cos`, `sin`, and the output slice are in global
memory. Each thread reads `K` interpolated samples; neighbouring pixels read
nearby detector positions, so locality is good and the kernel is
**bandwidth-bound** — the regime GPUs dominate. Production code binds `filtered`
to a **texture** so the hardware sampler does the interpolation for free.

**Independence.** Pixels are independent: no shared memory, no atomics, no
sync — the canonical tomographic kernel. The 3-D tomogram is a *stack* of these
independent slices (trivially parallel across slices too).

**CPU/GPU parity (the key idiom).** The per-sample interpolation lives in **one**
`__host__ __device__` function (`wbp_core.h::sample_projection_hd`), included by
both the CPU reference and the kernel, so they run byte-identical float math. The
ramp *weight* is likewise one shared `ramp_weight_hd`, so the cuFFT ramp and the
CPU DFT ramp are the same filter. `cos`/`sin` are precomputed once on the host and
uploaded, so the GPU never uses `cosf` where the CPU uses `cos`.

## 5. Numerical considerations

- **Precision.** Reconstruction sums are single precision; the CPU reference DFT
  accumulates in `double` for a clean baseline, then stores `float`.
- **The ramp and noise.** `|ω|` amplifies high frequencies and noise, so we apply
  a raised-cosine (Hann) roll-off toward Nyquist — standard apodization in WBP.
- **Determinism.** Alignment uses integer-bin lags with a fixed scan order and a
  tie-break toward zero; the per-pixel back-projection sum is in a fixed tilt
  order with no cross-thread reduction. So **stdout is byte-reproducible**
  (PATTERNS.md §3) — integer shifts plus fixed-precision samples.
- **FFT vs spatial convolution.** A frequency-domain ramp is a *periodic*
  convolution; a spatial Ram-Lak is a *linear* one. They differ near row edges
  (wrap-around vs zero-pad). We make the CPU reference an explicit **DFT** with the
  *same* ramp weights so both ramp paths are the same operation, and verify on the
  interior bins to a documented `5e-2` tolerance (the edge bins are largely outside
  the reconstructed field of view).

## 6. How we verify correctness

`main.cu` runs two checks (see also the `[verify]` line on stderr):

1. **Back-projection parity.** `backproject_cpu` and `backproject_gpu` consume the
   *same* (cuFFT-filtered) sinogram and their slices are compared with
   `max_abs_err`. They agree to ~`1e-6`, far inside the `1e-3` tolerance (the only
   difference is float FMA contraction order).
2. **Ramp parity.** The cuFFT ramp filter is compared against the CPU DFT ramp on
   the interior detector bins (tolerance `5e-2`); they agree to ~`1e-6` there.

Beyond CPU==GPU agreement, the result is **physically meaningful**: the recovered
`estimated shifts` track the injected drift to within ~1 bin (alignment works),
and the reconstruction's bright spot is the central disc (back-projection works) —
so the pipeline actually inverts the (limited-angle) Radon transform, not just
"the two implementations agree".

## 7. Where this sits in the real world

Production cryo-ET is far richer than this 2-D slice:

- **Alignment.** IMOD's `etomo`/`tiltxcorr` tracks **gold fiducial beads** across
  the whole series and solves a global model for translation, rotation,
  magnification, and the tilt-axis position; **AreTomo2** does fiducial-free patch
  cross-correlation / projection matching, on the GPU, and also corrects
  beam-induced motion. We teach only the 1-D translational, integer-precision
  core — the cross-correlation that underlies the coarse pass.
- **Reconstruction.** WBP is the fast classic; iterative **SART**/SIRT (as in the
  **ASTRA Toolbox**, CUDA) trade compute for fewer limited-angle artifacts. Both
  are GPU-accelerated for full 3-D volumes (a stack of the slice kernel here),
  often multi-GPU.
- **Missing wedge.** The ±60° limit leaves the wedge of unmeasured Fourier space;
  **IsoNet** (a PyTorch CNN) learns to restore isotropic resolution post hoc.

The two computational hearts you see here — the cross-correlation alignment and
the back-projection gather — are exactly the kernels those production tools
accelerate.

## References

- Kak & Slaney, *Principles of Computerized Tomographic Imaging* (1988) — Radon/FBP.
- Mastronarde (1997, 2005) — IMOD tilt-series alignment with fiducials.
- Zheng et al. (2022) — *AreTomo*: GPU fiducial-free alignment + reconstruction.
- van Aarle et al. (2015, 2016) — the **ASTRA Toolbox** GPU reconstruction algorithms.
- Liu et al. (2022) — *IsoNet*: deep-learning missing-wedge correction.
- NVIDIA cuFFT documentation — batched R2C/C2R transforms.
