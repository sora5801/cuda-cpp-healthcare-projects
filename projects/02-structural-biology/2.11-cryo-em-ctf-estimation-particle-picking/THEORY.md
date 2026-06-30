# THEORY — 2.11 Cryo-EM CTF Estimation (cuFFT defocus fit)

> A reduced-scope, didactic version of the CTF-estimation stage of single-particle
> cryo-EM processing (the job done by CTFFIND4 and RELION's CtfFind). We estimate
> the microscope **defocus** by fitting a parametric CTF model to a micrograph's
> radial power spectrum. Read this alongside the heavily-commented source; the code
> tour in [`README.md`](README.md) gives the reading order.

---

## The science

A transmission electron microscope does **not** record a faithful image of the
specimen. Cryo-EM specimens are *weak phase objects*: they shift the phase of the
electron wave by a tiny amount proportional to the projected Coulomb potential. To
turn that invisible phase shift into recordable amplitude contrast, microscopists
**deliberately defocus** the objective lens. Defocus, together with the lens's
**spherical aberration** (Cs), imposes an oscillating, sign-flipping transfer
function on spatial frequencies — the **Contrast Transfer Function (CTF)**.

The consequence is dramatic: some spatial frequencies are recorded with positive
contrast, some with negative contrast, and at the CTF's **zeros** no information is
transferred at all. Before you can average thousands of particle images into a 3-D
reconstruction, you must know the CTF of **every micrograph** so you can correct
for it (flip the negative bands back, weight by the envelope). The first step of
that is estimating the **defocus** of each micrograph — which is what this project
does.

You can *see* the CTF directly: the power spectrum of a micrograph shows bright
concentric rings, the **Thon rings**, sitting at the maxima of |CTF|². Their
spacing encodes the defocus. CTF estimation is, at heart, "measure the Thon-ring
spacing and read off the defocus."

---

## The math

### The contrast transfer function

For an untilted micrograph with no astigmatism (the teaching case), the CTF at a
spatial-frequency magnitude `k` (cycles/Å) is

```
CTF(k) = sin( χ(k) )                                                        (1)

χ(k)   = π·λ·Δz·k²  −  (π/2)·Cs·λ³·k⁴  +  φ                                 (2)
```

with

- `λ` — relativistic electron wavelength (Å); ≈ 0.0197 Å at 300 kV,
- `Δz` — **defocus** (Å), the unknown we estimate (under-focus is positive here),
- `Cs` — spherical aberration (Å); 2.7 mm = `2.7e7` Å,
- `φ = asin(amp_contrast)` — the constant phase from amplitude contrast.

A **power** spectrum records the squared Fourier amplitude, so the observable is

```
|CTF(k)|² = sin²( χ(k) )                                                    (3)
```

— always in [0, 1], with **Thon rings at its maxima** and dark gaps at its zeros
(where `χ = nπ`). The χ(k) phase grows like `k²` (defocus term), so the rings get
*closer together* at higher resolution; higher defocus ⇒ faster oscillation ⇒
tighter rings. That `k²`-vs-`k⁴` structure is exactly what the fit exploits.

These formulas live once in [`src/ctf_model.h`](src/ctf_model.h) as
`__host__ __device__` functions, shared verbatim by the CPU reference and the GPU.

### From micrograph to a 1-D profile

The 2-D power spectrum is `P(u,v) = |FFT(image)|²`. Because (for an untilted,
astigmatism-free image) the CTF depends only on `k = √(fu² + fv²)`, we **rotationally
average** `P` into a 1-D radial profile `prof[r]`, `r = 0 … N/2`. A smooth running-mean
background is subtracted so only the oscillating ring signal remains (the fit cares
about ring *positions*, not the falling envelope amplitude).

### The objective

We score a candidate defocus `Δz` by the **normalized cross-correlation** (Pearson
`r`) between the model `|CTF(k; Δz)|²` and the observed `prof[r]` over a fitting band
`[r_lo, r_hi)`:

```
NCC(Δz) =  Σ (mᵣ − m̄)(dᵣ − d̄)  /  √( Σ(mᵣ − m̄)² · Σ(dᵣ − d̄)² )           (4)
```

where `mᵣ = |CTF(k(r); Δz)|²` and `dᵣ = prof[r]`. NCC is invariant to the model's
and data's overall scale and offset, so we do **not** have to fit the (unknown)
envelope amplitude — only get the rings in the right place. The estimate is
`Δz* = argmax_Δz NCC(Δz)`. This is the same idea CTFFIND uses (it maximizes a
closely related cross-correlation over a 2-D grid including astigmatism).

---

## The algorithm

```
1. Remove the image mean (kill the DC term).
2. P = |FFT2(image)|²                     2-D power spectrum.
3. prof_raw[r] = mean of P over the ring   rotational average  (r = round(k_pixels)).
4. prof[r] = prof_raw[r] − runningmean(prof_raw)   background flattening.
5. for each candidate Δz in a grid:        defocus search
       score[Δz] = NCC(model(Δz), prof)    eq.(4)
6. Δz* = argmax score.
```

### Complexity (serial vs parallel)

| Stage | Serial cost | Parallel mapping |
|-------|-------------|------------------|
| 2-D FFT | naive DFT is **O(N⁴)**; FFT is **O(N² log N)** | cuFFT (one 2-D R2C plan) |
| Radial average | O(N²) over half-spectrum bins | one thread per bin, atomic scatter |
| Defocus search | O(`n_dz` · `nbins`) | **one thread per candidate defocus** |

For a real 4096 × 4096 micrograph the naive DFT (`N⁴ ≈ 2.8 × 10¹⁴` operations) is
utterly impractical — that single fact is why the FFT exists and why the GPU path
uses cuFFT. The defocus search is the embarrassingly-parallel part: hundreds of
candidate defoci, each scored independently.

---

## The GPU mapping

This project combines **two** of the repo's canonical GPU patterns (see
`docs/PATTERNS.md`).

### Pattern 1 — use a library kernel without it being a black box (cuFFT)

The 2-D FFT is a solved problem with a superb GPU library, **cuFFT**.
[`src/kernels.cu`](src/kernels.cu) documents exactly what `cufftPlan2d` +
`cufftExecR2C` compute:

```
X[v,u] = Σ_x Σ_y  img(x,y) · exp(−2πi (u·x + v·y) / N)
```

— the same 2-D DFT the CPU reference does by hand, in O(N² log N) instead of O(N⁴).
**R2C** (real-to-complex) exploits the Hermitian symmetry of a real image's FFT
(`X[−u,−v] = conj(X[u,v])`) and stores only the non-redundant half: `N/2+1` columns
per row. Hand-rolling this would mean a mixed-radix Cooley–Tukey FFT with
bit-reversal and twiddle tables — exactly the kind of thing a library should own.
We add only a tiny `power_kernel` for `|X|²` (one thread per complex bin).

### Pattern 2 — independent jobs + constant memory (the defocus search)

The search is **one thread per candidate defocus**. Every thread reads the *same*
observed radial profile, so we place that profile in **constant memory**
(`__constant__ double c_profile[]`), whose broadcast cache is ideal for data read
by every thread but never written — the same trick as the query fingerprint in the
1.12 Tanimoto flagship. Thread `i` computes `NCC(Δzᵢ)` via the shared
`ncc_model_vs_profile()` and writes `scores[i]`; the host takes the argmax (a tiny
`n_dz`-long reduction, not worth a second kernel).

### Memory hierarchy used

| Space | What lives there | Why |
|-------|------------------|-----|
| Global | image, half-spectrum, power, ring sums/counts, score curve | large arrays |
| **Constant** | the observed radial profile during the search | read by all threads, broadcast cache |
| Registers | per-thread `χ`, `NCC` accumulators | hot scalar math |

### Radial average — deterministic atomics

Many threads scatter into the same ring bin, so the radial average uses
`atomicAdd`. Floating-point `atomicAdd` is **non-associative** — the result would
depend on the (nondeterministic) thread order. So we accumulate in **64-bit integer
fixed-point** (`power × 256`, rounded) plus an integer ring count: integer adds
commute, making the sum **bit-reproducible across runs and identical to the CPU's
ordered sum** (PATTERNS.md §3). This is why the demo's stdout is byte-stable.

---

## Numerical considerations

- **Precision split.** cuFFT here is **single** precision (`cufftReal`/`cufftComplex`
  are `float`); the CPU reference DFT and all the fitting math are **double**. The
  radial profiles therefore differ at the ~1% level — a real, teachable consequence
  of FP32 vs FP64 transforms, not a bug.
- **Determinism.** The radial-average reduction uses integer fixed-point atomics
  (above); the defocus-search argmax uses strict `>` so the **first** (lowest-index)
  maximum wins on ties on both CPU and GPU. Result → stdout is identical every run.
- **DC removal.** The image mean is subtracted before the FFT (on both paths) so the
  enormous `|X(0,0)|²` term does not swamp the rings; the fit band also starts at
  `r_lo = 4` to skip the residual low-frequency spike.
- **NCC robustness.** Normalized cross-correlation removes the need to match the
  unknown envelope amplitude and is bounded in [−1, 1], so scores are comparable
  across candidates and numerically stable.

## How we verify correctness

`main.cu` runs **both** paths and checks three things, each to its honest tolerance
(PATTERNS.md §4):

1. **Exact** match of the recovered best-defocus *index* — the headline answer.
2. NCC at the winning candidate agrees to `< 5e-3` (the fit quality at the answer is
   robust: both transforms see the same strong ring signal there).
3. The full NCC curve agrees to `< 5e-2`, a **documented single-vs-double-FFT
   tolerance** — explicitly *not* machine precision, because comparing a FP32 cuFFT
   against a FP64 DFT cannot be bit-identical and we refuse to pretend otherwise.

A second, **scientific** check beyond CPU==GPU: the synthetic sample embeds a known
defocus (15000 Å), and the fitter recovers 15300 Å — within ~3 grid steps, validating
that the physics (not just the agreement) is right.

Edge cases handled: empty ring bins (count 0 → profile 0), degenerate fit bands
(`m ≤ 1` → impossible score), and a flat model/data (zero variance → score −2).

## Where this sits in the real world

This is a **teaching-scale** CTF estimator. Production tools differ as follows:

- **Astigmatism.** Real lenses have *two* defocus values along orthogonal axes
  (`Δz₁`, `Δz₂`, and an azimuth `α`). CTFFIND4 and RELION fit all three by scoring a
  **2-D** model against the **2-D** power spectrum (no rotational averaging), a 3-D
  search instead of our 1-D defocus scan. The k²/k⁴ machinery is identical; only the
  geometry grows.
- **Refinement.** After the grid search, real fitters refine with a continuous
  optimizer (Newton/Powell) and report a resolution cutoff (where ring detection
  fails), plus per-frequency phase/amplitude for correction.
- **Better spectra.** Production uses **tiled, windowed periodograms** (average many
  overlapping FFT patches) to beat down noise, and may apply spectral whitening.
- **Particle picking** (the catalog's second half) is a separate stage: template
  matching by FFT-based cross-correlation, or — today — a **GPU CNN** (TOPAZ, crYOLO)
  that runs inference on tiled micrographs. We focus on CTF estimation; picking is
  left as an exercise and described here for context.
- **Scale.** A facility fits **thousands** of 4k–8k micrographs per session, each
  over a 2-D defocus/astigmatism grid — precisely the throughput the GPU buys, with
  multi-stream processing across micrographs.

### Prior art

- **CTFFIND4** (Rohou & Grigorieff 2015) — the reference fast CTF estimator.
- **RELION CtfFind / gCTF** — GPU CTF estimation inside the RELION pipeline.
- **TOPAZ, crYOLO** — deep-learning particle pickers (the picking half of 2.11).
