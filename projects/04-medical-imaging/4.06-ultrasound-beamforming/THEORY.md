# THEORY — 4.6 Ultrasound Beamforming (Delay-and-Sum)

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

Medical ultrasound forms images from **sound echoes**. A hand-held probe holds a
linear row of tiny piezoelectric **elements** (64 here, 128–256 clinically).
The probe transmits a short high-frequency pulse (a few MHz) into the body; that
pulse reflects off **scatterers** — tissue boundaries, blood-cell clusters,
organ walls — and the echoes travel back. Each element converts the pressure
wave it receives into a voltage trace sampled in time: the **RF data**.

The catch: an element's raw trace is **not** an image. A scatterer at a given
point is at a slightly different distance from each element, so its echo arrives
at each element at a slightly different *time*. The raw traces are a jumble of
overlapping echoes from everywhere in the field. **Beamforming** is the
signal-processing step that turns those per-element time traces into a focused
spatial image — it decides, for each image point, "what echo came from *here*?"

The clinical payoff is real-time, radiation-free, cheap imaging: obstetrics,
cardiology (echocardiography), vascular flow (Doppler), point-of-care. The
bottleneck to *real-time* and to *3-D* imaging is the sheer arithmetic of
beamforming, which is exactly why the GPU matters.

## 2. The math

**Geometry** (SI units). The probe lies on the line `z = 0`. Element `e` sits at
`x_e = (e − (N−1)/2)·p` for `e = 0…N−1`, where `N = n_elements` and `p = pitch`.
The image is a grid in the `(x, z)` plane (`x` lateral, `z` depth ≥ 0); pixel
`(ix, iz)` is at `P = (x, z) = (x_min + ix·dx, z_min + iz·dz)`.

**Travel time.** We model a single **virtual transmit** from the array centre
`(0,0)` (a common teaching simplification; real systems transmit focused or
plane waves — see §7). The echo from pixel `P` recorded by element `e` traverses

$$ d_{tx}(P) = \lVert P - (0,0)\rVert, \qquad d_{rx}(P,e) = \lVert P - (x_e,0)\rVert, $$

$$ \tau_e(P) = \frac{d_{tx}(P) + d_{rx}(P,e)}{c}, $$

where `c` is the speed of sound (1540 m/s in soft tissue). `τ_e(P)` is the
round-trip **time of flight**: transmit-centre → `P` → element `e`.

**Delay-and-sum.** Let `s_e(t)` be element `e`'s RF signal. The beamformed value
at `P` is the coherent sum of every element's signal sampled at *its own* delay:

$$ b(P) = \sum_{e=0}^{N-1} w_e \; s_e\!\big(\tau_e(P)\big). $$

with apodization weights `w_e` (we use `w_e = 1`). Sampled RF lives at discrete
times `t_i = t_0 + i/f_s` (`f_s` = sampling rate, `t_0` = window start), so the
continuous time `τ_e(P)` maps to a **fractional sample index**

$$ \phi = (\tau_e(P) - t_0)\, f_s, \qquad
   s_e(\tau_e) \approx (1-f)\,s_e[i_0] + f\,s_e[i_0+1], $$

with `i_0 = ⌊φ⌋` and `f = φ − i_0` (linear interpolation). **B-mode brightness**
is the envelope `|b(P)|` (magnitude of the coherent sum).

**Why it focuses.** If a true scatterer sits at `P`, every `s_e(τ_e(P))` samples
the *same* physical echo at its peak → the terms add in phase (large `|b|`). If
`P` is empty, the terms are uncorrelated → they largely cancel (small `|b|`).
Delay-and-sum is, literally, "align the echoes in time, then add."

## 3. The algorithm

```
for each pixel P = (ix, iz):           # nx * nz pixels
    acc = 0
    for each element e:                # n_elements
        tau  = (d_tx(P) + d_rx(P,e)) / c
        phi  = (tau - t0) * fs
        i0   = floor(phi)
        if 0 <= i0 < n_samples-1:
            acc += interp(s_e, i0, phi-i0)
    image[ix,iz] = acc                 # signed coherent sum
```

**Complexity.** Serial work is `O(nx · nz · n_elements)` multiply-accumulates —
each pixel does `O(n_elements)` independent work, and there are `nx·nz` pixels.
For our sample `96·96·64 ≈ 5.9·10⁵` MACs; a clinical `512·512·128 ≈ 3.4·10⁷` per
**frame**, ×thousands of frames/s ≈ `10¹⁰–10¹¹` MAC/s. The **parallel depth** is
`O(n_elements)` (the inner sum) while the **parallel width** is `nx·nz` fully
independent pixels — an almost ideal data-parallel problem. Arithmetic intensity
is modest (a few FLOPs per RF byte read), so the kernel is **gather/bandwidth
bound**, not compute bound.

## 4. The GPU mapping

**Thread-to-data mapping.** One thread per output pixel. We launch a 2-D grid of
`16×16` blocks; thread `(blockIdx, threadIdx)` owns pixel

```
ix = blockIdx.x * blockDim.x + threadIdx.x      # lateral (fast axis)
iz = blockIdx.y * blockDim.y + threadIdx.y      # depth
```

Each thread runs the inner element loop (`das_pixel` in `beamform.h`) and writes
one image cell. No shared memory, no atomics, no synchronization — pixels are
independent.

```
   image (nx x nz)                         grid of 16x16 blocks
   +-------------------+                    +----+----+----+ ...
 z |  . . . . . . . .  |   one thread       |B00 |B10 |B20 |
 | |  . . .[P]. . . .  | <-- per pixel P     +----+----+----+
 v |  . . . . . . . .  |                     |B01 |B11 | .. |
   +-------------------+                     +----+----+----+
        x ->                          thread(tx,ty) in B(bx,by) -> pixel(ix,iz)
```

**Why this layout.** `threadIdx.x` strides over `ix` (the lateral image axis),
so 32 threads in a warp write 32 *consecutive* `image[iz*nx + ix]` cells →
**coalesced stores**. The RF reads are the interesting part: neighbouring pixels
have *similar* delays, so they read *nearby* RF samples — there is strong spatial
locality the L2 cache exploits even though we do not hand-tile it.

**Memory hierarchy.**
- **Kernel-parameter / constant space:** the small `BeamformGeom` struct is
  passed **by value**. Every thread reads `fs, c, pitch, …` from constant-banked
  parameter memory with broadcast — zero global traffic for scalars. (The
  catalog suggests loading *element positions* into **shared memory**; here
  positions are a one-line analytic formula `x_e = (e−(N−1)/2)·p`, cheaper to
  recompute than to stage — a deliberate teaching simplification. With irregular
  2-D/3-D arrays you *would* stage element coordinates in shared memory, as
  noted, and Exercise-style.)
- **Global memory:** only the bulky RF array (`n_elements·n_samples` floats).
  Read-only and `__restrict__`-qualified so the compiler keeps interpolated
  loads in registers.
- **Registers:** the per-pixel accumulator and delay scalars.

**Occupancy.** `16×16 = 256` threads/block is a solid default on sm_75…sm_89;
the kernel uses few registers and no shared memory, so occupancy is high and the
launch is bandwidth-limited rather than occupancy-limited.

**Where a CUDA library would fit (no black boxes).** The catalog mentions
**cuBLAS** for the element-weighted summation: if you precompute, per pixel, the
interpolated sample vector `s = [s_0(τ_0), …, s_{N-1}(τ_{N-1})]`, then the
beamformed value is the inner product `w·s` — a `gemv`/`dot`. In practice the
*delays and interpolation* (not the sum) dominate, and they are pixel-specific,
so a hand-written kernel that fuses delay+interp+sum (what we do) is simpler and
faster than building the `s` matrix to hand to cuBLAS. We therefore write the
sum by hand and explain the cuBLAS alternative rather than hide behind it
(CLAUDE.md §6.1.6). Likewise, the interpolation can be delegated to **texture
hardware** (`tex1Dfetch` with `cudaFilterModeLinear` gives free linear interp);
that is Exercise 3.

## 5. Numerical considerations

- **Precision: FP32.** RF data and beamforming are FP32 in real scanners; we
  match that. The accumulator is FP32 too, which is fine for `N ≤ a few hundred`
  terms of similar magnitude.
- **No atomics, no races.** Each pixel is written by exactly one thread, so there
  is no contention and **nothing reorders** within a pixel's sum. The element
  loop runs in the same order (`e = 0…N−1`) on CPU and GPU.
- **The only CPU↔GPU divergence is FMA.** The device may contract `a*b + c` into
  a single fused multiply-add (one rounding) where the host compiler emits a
  multiply then an add (two roundings). Over a sum of up to a few hundred terms
  this produces a tiny absolute difference (`~1.5·10⁻⁴` here). It is **not**
  nondeterminism — each side is bit-reproducible run to run; they just round
  slightly differently. That is exactly why we verify to `1e-3`, not `== 0`.
- **`floorf` vs `std::floor`.** Both pick the same bracketing integer sample, so
  the *interpolation stencil* is identical on both sides; only the final FMA
  rounding differs.

## 6. How we verify correctness

Two independent checks:

1. **GPU == CPU (independent implementations).** `reference_cpu.cpp` runs the
   serial triple loop; `kernels.cu` runs the parallel kernel. **Both call the
   exact same `das_pixel()` from `beamform.h`** (the `__host__ __device__` core,
   PATTERNS.md §2), so agreement is a strong check that the *parallelization* —
   indexing, bounds, memory traffic — is correct. We report `max_abs_err` and
   require it `≤ 1e-3`. We see `1.5e-4` (the FMA gap of §5).
2. **Recovered focal spot (the science).** The synthetic sample embeds **one
   point scatterer at a known `(4.0, 20.0) mm`**. A correct beamformer must focus
   all element echoes back there, so the **brightest pixel** must land on the
   scatterer. The demo reports `(3.9, 20.1) mm` — within one pixel
   (`dx≈0.21 mm`, `dz≈0.37 mm`). This validates that the *physics* is right, not
   just that two implementations agree.

**Determinism for the demo.** The deterministic image summary (focal spot, peak,
lateral profile) goes to **stdout** and is diffed against `expected_output.txt`;
the run-varying timings and `max_abs_err` go to **stderr** (shown, not diffed).
See PATTERNS.md §3.

## 7. Where this sits in the real world

Production beamforming is richer than this teaching DAS:

- **Transmit schemes.** Real systems do not use one virtual point transmit. They
  fire **focused beams** scanline-by-scanline, or **plane waves** / **diverging
  waves** for ultrafast imaging (thousands of frames/s), then **compound**
  multiple steered transmits. PICMUS is built around plane-wave RF. The receive
  delay model changes accordingly (the transmit-leg distance becomes a plane- or
  spherical-wave travel time).
- **Dynamic receive focusing & apodization.** The receive aperture and weights
  `w_e` vary with depth (f-number control) to keep resolution uniform and
  suppress side-lobes.
- **Adaptive beamformers.** *Coherence factor* (CF) and *delay-multiply-and-sum*
  (DMAS) add per-pixel statistics to suppress clutter; *minimum-variance* (MV /
  Capon) solves a small per-pixel linear system for optimal weights. All remain
  per-pixel-parallel — they bolt onto this kernel (Exercise 2).
- **Fourier-domain methods.** *f-k migration* and Stolt mapping beamform in the
  spatial-frequency domain via FFTs (cuFFT), trading the gather for transforms —
  asymptotically cheaper for plane-wave data.
- **Envelope & display.** Real pipelines add quadrature/Hilbert envelope
  detection, log-compression to dB, and scan conversion to screen geometry.

Tools to study: **MUST** (reference DAS + GPU, MATLAB), **Field II** (the
standard RF simulator), **k-Wave** (full-wave acoustic simulation, with a CUDA
solver), and open CUDA beamforming repos. Our point-scatterer simulator and
single-transmit DAS are the legible core they all build on.

---

## References

- Thomas L. Szabo, *Diagnostic Ultrasound Imaging: Inside Out* — the standard
  textbook; chapters on beamforming and array imaging.
- Jensen, *Field II* (<https://field-ii.dk/>) — the RF simulation model behind
  most academic ultrasound work; read it for the linear acoustics forward model.
- Garcia, *MUST — Matlab UltraSound Toolbox* (<https://www.biomecardio.com/MUST/>)
  — clean reference `das()` implementation and GPU notes.
- Montaldo et al., "Coherent plane-wave compounding…" (IEEE UFFC, 2009) — the
  ultrafast plane-wave imaging foundation that PICMUS evaluates.
- PICMUS challenge (<https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/>)
  — standard RF datasets and metrics for comparing beamformers.
