# THEORY — 4.10 Super-Resolution Microscopy Reconstruction

> For a reader who knows C++ but is new to CUDA and to single-molecule
> microscopy. See [README.md](README.md) for the tour and build. _Educational only._

## 1. The science

A conventional fluorescence microscope cannot resolve two point sources closer
than roughly **half the wavelength of light** — the **diffraction limit**, about
200–250 nm. A single fluorophore does not appear as a point but as a blurred spot,
the microscope's **point-spread function (PSF)**, well approximated by a 2D
Gaussian ~250 nm wide. If many fluorophores are lit at once (as in ordinary
staining) their PSFs overlap into an unresolvable haze.

**SMLM** (STORM, PALM, and relatives) defeats this with a trick of *time* rather
than optics. Using photoswitchable dyes, only a **sparse, random subset** of
fluorophores is made to "blink" on in each of thousands of camera frames. In a
sparse frame the individual PSFs are well separated, so each blob can be assigned
a **sub-pixel centre** far more precisely than its own width — down to ~10–20 nm,
because the *centroid* of a blob of N photons has an uncertainty ~PSF-width/√N.
Overlay the pin-point centres from every frame and a super-resolution image
emerges. STORM/PALM won the 2014 Nobel Prize in Chemistry precisely for this idea.

The three compute stages — **detect** candidate blobs, **localize** each to
sub-pixel accuracy, **render** all localizations into a fine image — are what this
project implements on the GPU.

## 2. The math

A camera frame is a pixel grid `I(r,c)`. An isolated on-emitter centred at
sub-pixel position `(x₀, y₀)` with integrated intensity `A` and background `b`
contributes the PSF

```
I(r,c) = b + (A / (2πσ²)) · exp( −((c−x₀)² + (r−y₀)²) / (2σ²) )
```

where `σ` is the PSF width in pixels (~1–1.5 px for typical optics), `c` is the
column (x) and `r` the row (y). **Localization** is the inverse problem: given the
noisy pixels of one blob, estimate `(x₀, y₀)` (and optionally `A`, `σ`, `b`).

The gold-standard estimator is **maximum likelihood** under a Poisson (photon
shot-noise) model, or nonlinear least squares — both iterative fits of the five
parameters. This project uses the simpler, robust **Gaussian-weighted centroid**
estimator: the intensity-weighted mean position, iteratively re-weighted by a
Gaussian window centred on the current estimate. For an isolated symmetric PSF the
weighted centroid converges to `(x₀, y₀)`; it is the fast method ThunderSTORM
offers and the historical workhorse of particle tracking.

**Rendering** places each localization `(xᵢ, yᵢ)` with weight `Aᵢ` into a
super-resolution grid upsampled by `U=8`: bin `(⌊yᵢU⌋, ⌊xᵢU⌋)` accumulates `Aᵢ`.
The reconstructed image is `S(r',c') = Σᵢ Aᵢ · 1[bin(i) = (r',c')]`.

## 3. The algorithm

```
for each frame f:
    for each interior pixel (r,c):                     # DETECT
        if I(r,c) > threshold and I(r,c) is a strict 3x3 local max:
            (x,y) <- intensity-weighted centroid of the 7x7 patch (bg-subtracted)
            repeat FIT_ITERS times:                    # LOCALIZE (refine)
                w_i   = exp(-|p_i-(x,y)|^2 / 2σ²) · I_i    (I_i = pixel - bg, ≥0)
                (x,y) <- Σ w_i p_i / Σ w_i
            emit localization (x, y, photons, σ, f)
for each localization:                                 # RENDER
    add photons into super-resolution bin (⌊yU⌋, ⌊xU⌋)
```

**Complexity.** Let `P = F·H·W` be the total pixels and `L` the number of
localizations. Detection is `O(P)` (a constant-size 3×3 test per pixel).
Localization is `O(L · FIT_ITERS · 49)` — a few hundred flops per emitter, all in
registers. Rendering is `O(L)`. The whole pipeline is linear in the data, and both
the per-pixel detect+fit and the per-localization render are **fully parallel**
(no dependence between pixels or between emitters).

## 4. The GPU mapping

Two kernels, mirroring the two parallel loops:

**`localize_kernel` — one thread per interior pixel (of one frame).** Thread `t`
maps to interior pixel `(r,c) = (t/IW + PATCH_R, t%IW + PATCH_R)` where
`IW = W − 2·PATCH_R`. It runs the *same* `smlm_is_local_max` + `smlm_localize` the
CPU uses, reading its own 7×7 patch from global memory and writing one
`Localization` into **output slot `t`** plus a `valid` flag. Because a slot's index
*is* its scan position, gathering the valid slots in index order reproduces the
CPU's canonical `(frame, row, col)` ordering for free — so the two localization
lists are comparable element-for-element. Launch: `block = 256`,
`grid = ⌈interior/256⌉`. No shared memory or atomics: the fits are independent.

**`render_kernel` — one thread per localization.** Thread `i` computes its
super-resolution bin and does one `atomicAdd` of its **fixed-point** photons into
the shared image. Many emitters can fall in the same bin (that pile-up *is* the
image), so the writes collide → atomics.

```
frame (H x W)                         localize_kernel                super-res image
+---------------------+     one thread per interior pixel        (H*U) x (W*U)
| . . . o . . . . . . |     -> reads its 7x7 patch               +-----------------+
| . . . . . . o . . . |     -> if local-max: fit (x,y)           |   .  *          |
| . . o . . . . . . . |  ==========================>  render     |      *          |
| . . . . . . . o . . |     one thread per localization          |   *      *      |
+---------------------+     -> atomicAdd photons to its bin       +-----------------+
```

**Memory hierarchy.** The patch lives in **registers** during the fit (the loop
bounds are compile-time `constexpr`, so the compiler can fully unroll). The frame
is read from **global** memory; neighbouring threads read overlapping patches, so a
production kernel would stage the patch in **shared** memory and use one *warp* per
candidate (Exercise 1). The render's image is global; a privatized shared-memory
histogram (Exercise 4) would cut atomic traffic on dense regions. No CUDA library
is needed here — the fit is a small custom kernel and the render is a hand-written
atomic scatter; the catalog's cuFFT appears only in the SIM branch (§7), which this
reduced-scope version omits.

## 5. Numerical considerations

- **Precision.** The fit accumulates in **`double`**. It is cheap (a few hundred
  flops per emitter) and doubles keep the weighted centroid well-conditioned, so
  precision is essentially free here.
- **Determinism of the fit.** `smlm_localize` is a *fixed* iteration count of the
  same arithmetic in the same order on both CPU and GPU — no reduction, no atomics,
  no data-dependent branching beyond the clamp `I≥0`. `exp()` on the same `double`
  argument is identical under IEEE-754 on host and device. So the fits are
  bit-identical (mean error `0` in the demo).
- **Determinism of the render (the key trick).** Float `atomicAdd` is **not
  associative**, so summing many emitters into a bin in nondeterministic thread
  order would change the last bits → irreproducible and CPU≠GPU. We instead add
  each emitter's intensity as a **fixed-point integer** (`smlm_to_fixed`, scale
  `2¹⁶`). Integer atomic adds **commute**, so the image is identical regardless of
  order and matches the CPU exactly. This is the atomic-reduction determinism
  lesson of docs/PATTERNS.md §3 (same idea as projects 5.01 and 11.09).
- **Detection ties.** A strict `>` local-max test avoids double-firing on a PSF's
  shoulder; an exact intensity plateau (measure-zero on noisy data) yields no
  detection, which we accept for determinism.

## 6. How we verify correctness

`main.cu` runs the entire pipeline on CPU (`detect_and_localize_cpu` +
`render_image`) and GPU (`smlm_gpu`) and checks, in increasing strength:

1. **Same localization count** — the detect + fit produced the same emitters.
2. **Exact image checksum + bright-bin count** — the fixed-point render summed to
   the same total in the same number of bins (`== 0` difference).
3. **Every fixed-point pixel identical** — no two bins swapped contents.
4. **Mean statistics within `1e-6`** — means of thousands of doubles computed in
   the same order; the slack is generous for last-ULP differences (docs/PATTERNS.md
   §4). In practice the demo reports `mean_err = 0`.

Because the GPU and an *independent* serial implementation agree to the bit on the
integer/render fields, we trust the GPU. Beyond CPU/GPU parity, the result is
physically sensible: the mean fitted PSF width (~1.35 px) matches the `σ = 1.3 px`
the synthetic data was generated with, and the 187 localizations reconstruct the
two ground-truth lines — validating the science, not just the port.

## 7. Where this sits in the real world

Production SMLM does much more than this teaching pipeline:

- **The fit.** Real localizers (ThunderSTORM, DECODE, SMAP, picasso) use iterative
  **maximum-likelihood** or least-squares 2D-Gaussian fitting (amplitude, x, y, σ,
  background), often with an **sCMOS noise model** and **multi-emitter** fitting to
  resolve overlapping blobs at higher densities. **DECODE** replaces the per-blob
  fit with a trained neural network — orders of magnitude faster and more accurate
  at high density.
- **Rendering.** Localizations are usually **splatted as Gaussians** (weighted
  across neighbouring bins by their localization uncertainty), not hard-binned, and
  corrected for **sample drift** over the long acquisition.
- **Other super-resolution families** in the catalog use entirely different math on
  the GPU: **SRRF/SOFI** compute radial-gradient fluctuations or temporal cumulants
  per pixel over the time stack (`O(N·T)`); **SIM** reconstruction does per-
  orientation/phase **FFTs** (cuFFT) and **OTF/Wiener** inversion in k-space. Those
  are natural follow-on projects; this one teaches the localization-and-render core
  that STORM/PALM share.

This is a deliberately **reduced-scope teaching version** (CLAUDE.md §13): the
weighted-centroid fit + hard-bin render make the GPU pattern (independent jobs +
atomic reduction) and the CPU/GPU determinism story crisp, while the paragraphs
above map the path to a research-grade pipeline.

---

## References

- Betzig et al. (2006), *Imaging Intracellular Fluorescent Proteins at Nanometer
  Resolution* (PALM); Rust, Bates & Zhuang (2006), *STORM* — the founding methods.
- Thompson, Larson & Webb (2002) — the √N localization-precision result.
- Ovesný et al. (2014), *ThunderSTORM* — the open-source localizer; documents the
  weighted-centroid and MLE fits used here and in production.
- Speiser et al. (2021), *DECODE* — deep-learning SMLM localization (the catalog's
  starter repo).
- Gustafsson et al. (2016), *NanoJ-SRRF*; fairSIM docs — the SRRF and SIM branches.
- NVIDIA CUDA C++ Programming Guide — atomics; docs/PATTERNS.md §1–§3 — the pattern.
