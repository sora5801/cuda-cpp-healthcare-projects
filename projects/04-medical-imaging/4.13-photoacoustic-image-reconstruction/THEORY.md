# THEORY — 4.13 Photoacoustic Image Reconstruction

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

**Photoacoustics** couples light in and sound out. Fire a nanosecond laser pulse into tissue.
Molecules that absorb that wavelength — hemoglobin in blood, melanin, or an injected dye — convert
the light to a tiny, near-instantaneous temperature rise (millikelvin). The heated region expands
faster than sound can escape it (the *stress-confinement* regime), so it launches a pressure
transient: an ultrasound pulse. This is the **photoacoustic effect** (Bell, 1880; revived for
imaging in the 1990s).

The clinical appeal is that you get **optical contrast** (what absorbs light — e.g. oxygenated vs
deoxygenated blood, hence blood-oxygen mapping) at **ultrasound resolution and depth** (centimeters,
far deeper than pure optical microscopy, because sound scatters ~1000× less than light in tissue).
Applications: breast-tumor angiogenesis, skin melanoma depth, small-animal whole-body imaging, and
image-guided intervention.

We do not measure the absorber map directly. We measure **pressure vs time** at an array of
ultrasound sensors around the object. **Reconstruction** is the inverse problem: from the recorded
traces, infer the initial pressure distribution `p₀(x)` — a proxy for "how much light was absorbed
where". That inverse problem is what this project solves, using the simplest robust method:
**delay-and-sum (DAS) backprojection**.

```
   laser pulse
       |                       sensor ring (records pressure vs time)
       v                     .  .  .  .  .
   [ tissue with           .              .
     absorbers ]         .    (o)   (o)     .     (o) = optical absorber
                        .        (o)         .    each emits an outgoing
                         .                  .     ultrasound wave at speed c
                           .              .
                             .  .  .  .  .
```

## 2. The math

**Forward model.** A point absorber at `q` emits a spherical acoustic wave traveling at the speed
of sound `c`. Sensor `s` at position `pₛ` records that wave delayed by the travel time

```
    τₛ(q) = |pₛ − q| / c            [seconds]
```

Its recorded trace `gₛ(t)` is (a superposition over all absorbers of) a short pulse centered at
`t = τₛ(q)`, scaled by the absorber strength and geometric spreading.

**Inverse problem (what we compute).** Given the traces `{gₛ(t)}` and the geometry `{pₛ}`, estimate
the source image `b(x)`. **Delay-and-sum backprojection** inverts the delay: for a candidate image
point `x`, the wave *from* `x` reaches sensor `s` at sample time `τₛ(x) = |x − pₛ|/c`. So we read
each sensor's trace at exactly that time and add the readings up:

```
    b(x) = (1/S) · Σ_{s=0}^{S−1}  gₛ( |x − pₛ| / c )
```

where `S = n_sensors`. If `x` is a true source, every sensor contributes its pulse *in phase* and
they reinforce (a bright peak). If `x` is empty, the contributions land at random phases and largely
cancel. `b(x)` is our reconstructed pixel value.

**Symbols and units** (all SI):

| Symbol | Meaning | Units |
|---|---|---|
| `x = (wx, wy)` | image point (pixel world coordinate) | m |
| `pₛ = (sxₛ, syₛ)` | sensor `s` position | m |
| `c` | speed of sound (≈1500 in soft tissue) | m/s |
| `dt` | sensor sample period | s |
| `gₛ(t)` | pressure trace of sensor `s`; sampled `sig[s·n_samples + t]` | arb. pressure |
| `τₛ(x)` | travel time from `x` to sensor `s` | s |
| `fidx = τₛ/dt` | fractional sample index into `gₛ` | samples |

Because `τₛ/dt` is almost never an integer, we **linearly interpolate** between the two neighbouring
samples `⌊fidx⌋` and `⌊fidx⌋+1` (function `pa_sample_trace` in `src/pa_core.h`). Samples that would
arrive outside the recorded window `[0, n_samples−1]` contribute 0.

**Continuous ancestor.** DAS is the discrete cousin of the **universal back-projection** formula
(Xu & Wang, 2005): the exact 2-D/3-D inversion additionally applies a time-derivative to each trace
and weights each sensor by the solid angle it subtends. We omit those refinements in the teaching
kernel and describe them in §7 — DAS alone already localizes point sources well.

## 3. The algorithm

```
load traces gₛ, sensor positions pₛ, geometry (c, dt, img, world_half)
for each pixel (px, py):                 # img × img pixels
    x ← world coordinate of (px, py)
    acc ← 0
    for each sensor s:                   # n_sensors sensors
        d    ← |x − pₛ|                  # Euclidean distance
        fidx ← (d / c) / dt              # fractional sample index
        acc  += interp(gₛ, fidx)         # linear interpolation, 0 if out of range
    b[px,py] ← acc / n_sensors
```

**Complexity.** `O(img² · n_sensors)` work in 2-D — one inner sensor loop per pixel. In 3-D it is
`O(img³ · n_sensors)`. The catalog's headline "68 billion operations" is exactly this product for a
`256³` volume and `1024` sensors. The **serial depth** per pixel is `O(n_sensors)`; the pixels are
mutually independent, so the *parallel* depth of the whole reconstruction is just `O(n_sensors)`
(one pixel's loop) if we had a processor per pixel — which is precisely what a GPU approximates.

**Data-access pattern.** Each pixel reads all `n_sensors` positions and one interpolated sample per
sensor. Neighbouring pixels read *nearly the same* trace samples (their delays differ by a fraction
of a sample), so there is heavy **data reuse** across the image — a cache-friendly gather. Arithmetic
intensity is modest (a sqrt, a few multiplies, two loads per sensor), so the kernel is
**memory/latency-bound**, not compute-bound, which shapes the GPU strategy below.

## 4. The GPU mapping

**Thread-to-data mapping.** The image is 2-D, so we use a **2-D thread grid**: thread `(px, py)`
owns output pixel `(px, py)` and runs the entire inner sensor loop for that one pixel. This is the
canonical **gather** pattern (docs/PATTERNS.md §1), identical in shape to CT filtered backprojection
(flagship 4.01) — the difference is only *what* is gathered (a time-series sample vs a detector
sample) and *how the index is computed* (a travel-time delay vs a ray-detector crossing).

**Launch configuration.** `block = 16×16 = 256 threads`; `grid = ⌈img/16⌉ × ⌈img/16⌉`. A square
tile maps naturally onto the square image and keeps spatially-neighbouring pixels — which read
overlapping trace samples — in the same block, so they hit the same L2/texture cache lines. 256
threads is a solid occupancy default on sm_75…sm_89 (8 warps to hide global-memory latency). Border
blocks contain out-of-range threads (`px≥img` or `py≥img`); they early-`return` so they never write
out of bounds.

```
        image (img × img)                     grid of 16×16 blocks
   px→ 0                 img-1
 py ┌───────────────────────┐         ┌────┬────┬────┬────┐
  0 │ . . . . . . . . . . . │         │ B  │ B  │ B  │ B  │  each block = 16×16 threads
    │ . . . . . . . . . . . │   ==>   ├────┼────┼────┼────┤  each thread = ONE pixel
    │ . . . . . . . . . . . │         │ B  │ B  │ B  │ B  │  thread (px,py) runs the
img-1 . . . . . . . . . . . │         └────┴────┴────┴────┘  full sensor loop for (px,py)
    └───────────────────────┘
```

**Memory hierarchy.**
- **Global memory** holds the sensor positions `d_sx/d_sy` (tiny) and the traces `d_sig` (the bulk).
  We keep them in plain global memory for clarity; the read-only, heavily-reused traces are served
  well by the L2 cache. Marking pointers `const … __restrict__` lets the compiler assume no aliasing
  and cache loads in registers.
- **Registers** hold each thread's accumulator `acc`, its pixel coordinates, and the precomputed
  reciprocals — the hot inner-loop state, so no shared memory is required.
- **No shared memory / no atomics.** Pixels never write to each other's outputs (one store per
  thread at the end), so this is a pure gather with zero synchronization — the easiest, fastest kind
  of GPU kernel.

**Production optimizations (named in the catalog, left as exercises).**
- **Texture memory** for the traces: a 1-D texture does the linear interpolation in *hardware* and
  caches spatially — the standard trick for DAS/CT backprojection (Exercise 1).
- **Constant memory / shared LUT** for sensor geometry: `pₛ` is read by every thread and never
  changes, a perfect fit for constant memory's broadcast cache or a per-block shared LUT.
- **cuFFT k-space propagation:** the *model-based* reconstruction (not DAS) solves the wave equation
  with a pseudospectral k-Wave method, whose per-step FFTs are cuFFT calls (see project 8.03 for the
  cuFFT idiom); multi-GPU builds decompose over k-space planes. That is a different algorithm class,
  summarized in §7.

## 5. Numerical considerations

- **Precision: FP32.** Ultrasound traces are ~12–14-bit ADC data; single precision is far more than
  the measurement warrants and doubles GPU throughput/halves bandwidth versus FP64. The distances and
  delays are `O(0.01–0.03 m)` / `O(1e-5 s)`, comfortably within FP32's ~7 significant digits.
- **Determinism.** The inner sum is a **fixed-order** loop over sensors `s = 0…S−1`, identical on CPU
  and GPU. There are no atomics and no cross-thread reductions, so **stdout is bit-for-bit
  reproducible** every run (timings, which vary, are printed to stderr — PATTERNS.md §3).
- **CPU vs GPU divergence (FMA).** The CPU and GPU call the *same* `pa_pixel_das` from `pa_core.h`,
  but nvcc **contracts** `a*b + c` into a single fused multiply-add (one rounding) while the host
  compiler rounds the multiply and the add separately. Over a ~64-term sum this accumulates to a
  `~3e-4` absolute difference. This is real, expected FMA behavior (PATTERNS.md §4), *not* a bug.
- **Interpolation & windowing.** Linear interpolation is `O(dt²)` accurate; out-of-window arrivals
  return 0, which is why the sample geometry is chosen so every relevant delay lands inside
  `[0, n_samples−1]` (see `scripts/make_synthetic.py`).

## 6. How we verify correctness

Two independent checks, following PATTERNS.md §4:

1. **GPU vs CPU agreement.** `src/reference_cpu.cpp` reconstructs the same image with an obvious
   serial double loop, and `main.cu` computes `max_abs_err = max |b_gpu − b_cpu|`. We require it
   ≤ **1e-3**. Why 1e-3 and not 0? As explained in §5, FMA contraction makes the two differ at the
   `~3e-4` level; with peak reconstructed values ≈ 33, that is a physically negligible ~0.003 %.
   Verifying to a small *physical* tolerance and saying so is more honest than pretending the two are
   bit-identical. Because the reference shares the per-pixel physics (`pa_core.h`) yet loops
   completely differently (serial vs 2-D grid), their agreement is strong evidence the parallel
   version is correct.
2. **Recovering the ground truth (the science check).** The synthetic sample plants three point
   absorbers at known locations, the strongest at the origin. A correct reconstruction must show its
   **brightest pixel at that origin** — and it does: `peak … at (px,py)=(47,47) = (x,y)≈(0,0) m`.
   This validates the *physics*, not just CPU==GPU arithmetic. (The peak pixel `(47,47)` is the grid
   node nearest `(0,0)`; the `0.1 mm` offset is pixel discretization, a nice teaching artifact.)

Edge cases handled: ragged border blocks (`px/py ≥ img` guarded), arrivals outside the recorded
window (return 0), and a malformed/truncated data file (the loader throws and the program exits 2).

## 7. Where this sits in the real world

This is a **reduced-scope teaching version**. Production PA reconstruction differs in several ways:

- **Universal back-projection** (Xu & Wang, 2005) is the exact analytic inversion: it applies a
  time-derivative `∂g/∂t` (or a ramp filter, as in CT) to each trace and weights sensors by solid
  angle, removing DAS's low-frequency blur and bipolar artifacts. **k-Wave** implements this and
  **time-reversal** (re-emit the recorded pressure backwards in time through a simulated medium),
  which is the gold standard for heterogeneous media.
- **Model-based iterative reconstruction** builds the forward operator `A` (a k-space pseudospectral
  wave solve, cuFFT per step) and solves `min ‖A p₀ − g‖² + λ R(p₀)` — enabling heterogeneous
  speed-of-sound `c(x)`, frequency-dependent acoustic attenuation, and regularization/compressed
  sensing from few sensors. This is where the GPU's cuFFT throughput and multi-GPU k-space
  decomposition matter, and where "quantitative PAI" comes from.
- **Deep learning** end-to-end networks now map raw traces directly to images, learning to suppress
  limited-view and sparse-aperture artifacts.
- **Bipolar pulses & envelopes.** Real PA pulses are bipolar (the Gaussian's derivative), so raw DAS
  images are bipolar; production takes the analytic-signal **envelope** (Hilbert magnitude) before
  display. We use a unipolar pulse so the peak-at-source check is unambiguous (Exercise 2).
- **Full 3-D, finite sensors, SNR.** Clinical systems reconstruct 3-D volumes from 2-D sensor arrays
  with finite-aperture directional sensors and real noise — the `256³ × 1024` regime that makes the
  GPU mandatory.

---

## References

- **M. Xu & L. V. Wang, "Universal back-projection algorithm for photoacoustic computed
  tomography," _Phys. Rev. E_ 71, 016706 (2005).** The exact inversion DAS approximates — read for
  the derivative + solid-angle weighting.
- **L. V. Wang & S. Hu, "Photoacoustic Tomography," _Science_ 335 (2012).** Accessible overview of
  the physics and clinical scope.
- **B. Treeby & B. Cox, "k-Wave: MATLAB toolbox for the simulation and reconstruction of
  photoacoustic wave fields," _J. Biomed. Opt._ 15, 021314 (2010).** The reference toolbox; study its
  forward simulation and time-reversal. CUDA fluid solver: https://github.com/klepo/k-Wave-Fluid-CUDA.
- **PyTomography** (https://github.com/lukepolson/pytomography) — a readable GPU tomography codebase;
  compare its backprojection implementation to `src/kernels.cu`.
- Flagship **4.01** (CT filtered backprojection) in this repo — the same gather pattern with a ramp
  filter; the closest sibling to study side-by-side.
