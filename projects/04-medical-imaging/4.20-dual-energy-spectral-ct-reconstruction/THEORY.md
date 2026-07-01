# THEORY — 4.20 Dual-Energy / Spectral CT Reconstruction

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## The science

### Why one CT number is ambiguous

A conventional CT scan measures, along each ray, how much an X-ray beam is
attenuated. The reconstructed image is a map of the **linear attenuation
coefficient** μ [1/cm]. The problem: **two different materials can have the same
μ at one energy.** Iodinated contrast, calcified plaque, and dense bone can all
present the *same* CT number (Hounsfield unit) even though they are chemically
very different. A single-energy scan simply cannot tell them apart.

### The dual-energy trick

Attenuation depends on photon energy, and it does so **differently for different
materials**. Two physical processes dominate in the diagnostic energy range
(~30–140 keV):

- **Photoelectric absorption** — scales steeply with atomic number
  (roughly `∝ Z^{3–4}`) and falls fast with energy (`∝ 1/E³`). High-Z materials
  like iodine (Z=53) and calcium (Z=20) show a big photoelectric signal at low
  energy.
- **Compton scattering** — nearly independent of Z, gently varying with energy.
  Dominant for soft tissue (low Z) and at high energy.

So if you scan the **same** ray with a **low-energy** spectrum (which weights the
photoelectric effect heavily) and again with a **high-energy** spectrum (which
weights Compton), the *ratio* of the two measurements encodes the material
composition. Two spectra → two numbers per ray → enough to solve for **two
unknowns**: the amounts of two chosen **basis materials**.

We use the classic basis pair: **material 1 = water / soft tissue** (low Z) and
**material 2 = iodine / bone** (high Z). Any tissue's attenuation is written as a
mixture of these two — the **basis-material model** of Alvarez & Macovski (1976).

### What "decomposition" produces

For each ray we recover two numbers:

- `t1` = the **water-equivalent path length** [cm] along the ray,
- `t2` = the **iodine-equivalent path length** [cm] along the ray.

From those you can synthesize a **virtual monoenergetic image** at any energy, do
**K-edge imaging** of a contrast agent, or feed the two "material sinograms" into a
tomographic reconstruction to get two co-registered material images. This project
computes the per-ray `(t1, t2)` — the step that turns two raw spectral
measurements into physically meaningful material path lengths.

---

## The math

### Monochromatic Beer–Lambert (the ideal case)

For a single photon energy `E`, the log-attenuation along a ray is

```
m(E) = -ln(I / I0) = ∫_ray μ(E, x) dx.                                    (1)
```

In the basis-material model the local attenuation splits into two materials whose
energy dependence is known:

```
μ(E, x) = c1(x)·μ1(E) + c2(x)·μ2(E).                                      (2)
```

`μ1(E)`, `μ2(E)` are the (tabulated) attenuation curves of the two basis
materials; `c1(x)`, `c2(x)` are their local amounts. Substituting (2) into (1),
the **energy** part factors out of the **spatial** integral, and the ray is fully
described by the two **path-length integrals**:

```
t1 = ∫_ray c1 dx      t2 = ∫_ray c2 dx.                                   (3)
```

If the beam were monochromatic, decomposition would be *linear*: two energies give
two linear equations `m(E_a) = μ1(E_a)·t1 + μ2(E_a)·t2`, etc. Solve the 2×2 linear
system and you are done.

### Polychromatic reality (why it is nonlinear)

Real tubes emit a **spectrum** `S_e(E)` (e = lo/hi), and the detector integrates
over energy. The measured log-attenuation is therefore

```
                 ∫ S_e(E) · exp( -(t1·μ1(E) + t2·μ2(E)) ) dE
 f_e(t1,t2) = -ln ───────────────────────────────────────────── .        (4)
                             ∫ S_e(E) dE
```

This is the log of a **spectrum-weighted average of exponentials** — the source of
**beam hardening** (low-energy photons are removed preferentially, so the beam
"hardens" and the effective μ drifts with path length). `f_e` is nonlinear in
`(t1, t2)`, so the inverse problem needs an iterative solver.

We discretize the energy integral as a sum over `K = NUM_ENERGIES` samples with
normalized weights `w_e[k]` (so the denominator `∫ S_e dE = 1`):

```
 f_e(t1,t2) = -ln( Σ_k w_e[k] · exp( -(t1·μ1[k] + t2·μ2[k]) ) ).          (5)
```

### The inverse problem

Given the two measurements `(m_lo, m_hi)` at a bin, find `(t1, t2)` with

```
 f_lo(t1,t2) = m_lo,     f_hi(t1,t2) = m_hi.                              (6)
```

Two nonlinear equations, two unknowns, per bin.

---

## The algorithm

### Newton's method for a 2×2 system

Write the residual vector `g(x) = (f_lo(x) − m_lo, f_hi(x) − m_hi)` with
`x = (t1, t2)`. Newton iterates

```
 x_{n+1} = x_n − J(x_n)^{-1} · g(x_n),                                    (7)
```

where `J` is the 2×2 **Jacobian** of `(f_lo, f_hi)` w.r.t. `(t1, t2)`. The partials
come from differentiating (5). Let `T = Σ_k w[k]·exp(−p_k)` with
`p_k = t1·μ1[k] + t2·μ2[k]`. Then

```
 ∂f/∂t1 = ( Σ_k w[k]·exp(−p_k)·μ1[k] ) / T   =  ⟨μ1⟩_eff,                 (8)
 ∂f/∂t2 = ( Σ_k w[k]·exp(−p_k)·μ2[k] ) / T   =  ⟨μ2⟩_eff.
```

Each partial is a **transmission-weighted mean** of the attenuation curve — the
"effective attenuation coefficient" for the current path. It *drifts* as the path
lengthens: that drift **is** beam hardening, and it is exactly why the linear
solution is only approximate and Newton is needed.

The 2×2 inverse is closed-form:

```
 J^{-1} = (1/det) · [  J11  −J01 ;  −J10   J00 ],   det = J00·J11 − J01·J10.
```

### The full per-bin recipe (implemented in `dect.h`)

```
1. Seed x0 = (t1,t2) from the LINEARISED problem (solve the 2×2 linear system
   using spectrum-averaged coefficients ⟨μ⟩; see reference_cpu.cpp::linear_init).
2. Repeat up to MAX_NEWTON_ITER:
     a. Evaluate f_lo, f_hi and their partials with one pass over the K energies.
     b. residual = max(|f_lo−m_lo|, |f_hi−m_hi|); if < NEWTON_TOL, stop.
     c. Assemble J, invert (guard tiny det), step x ← x − J⁻¹g.
     d. Clamp t1,t2 ≥ 0 (path lengths are physical).
3. Output (t1,t2) and the iteration count.
```

### Complexity

- **Per bin:** each Newton iteration is `O(K)` (one loop over energies for `f` and
  the partials). With `I ≈ 5–8` iterations that is `O(I·K)` — a few thousand FLOPs
  including `K` exponentials.
- **Whole sinogram (n bins):** `O(n·I·K)`. Crucially the `n` bins are **independent**
  — no coupling, no shared state — so the *parallel* depth is just `O(I·K)`, and the
  work spreads across as many threads as the GPU can run. This is the ideal shape
  for data-parallel hardware.

```
serial CPU:   for each of n bins:  Newton( O(I·K) )        →  O(n·I·K), one at a time
parallel GPU: n threads, each:     Newton( O(I·K) )        →  O(I·K) depth, n-wide
```

---

## The GPU mapping

### Thread-to-data mapping

One **thread per sinogram bin**. Thread `i = blockIdx.x·blockDim.x + threadIdx.x`
owns bin `i`, reads `(m_lo[i], m_hi[i])` from global memory, runs the entire Newton
solve in **registers**, and writes `(t1[i], t2[i], iters[i])`. A **grid-stride
loop** lets a fixed-size grid (≤ 1024 blocks × 128 threads) cover an arbitrarily
large sinogram (a real scan's ~10⁸ bins):

```
for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x)
    solve bin i;
```

There are **no data dependencies between bins**, so there is **no shared memory and
no atomics** — the cleanest possible parallel pattern (contrast with the
atomic-reduction pattern of flagship 11.09).

### Memory hierarchy — the design decisions

| Data | Where | Why |
|---|---|---|
| Spectral model (both spectra + both μ curves, ~960 B) | **`__constant__`** | Read by *every* thread, never written during the launch. The constant cache **broadcasts** one address to a whole warp in a single transaction — ideal for uniform read-only parameters. This is the same idea as the constant-memory query in flagship 1.12. |
| Measurements `m_lo[i]`, `m_hi[i]` | **global**, coalesced | Consecutive threads read consecutive bins → contiguous 128-byte transactions, full bandwidth. |
| Working state `t1, t2, J, g, …` | **registers** | The entire Newton solve is scalar per-thread work; keeping it in registers avoids any global traffic in the hot loop. |
| Outputs `t1[i], t2[i], iters[i]` | **global**, coalesced | One write per thread at the end. |

The `#pragma unroll` over the compile-time-constant `NUM_ENERGIES` turns the inner
energy loop into straight-line code (no loop overhead), and `exp`/`log` map to
hardware-accelerated device intrinsics.

### Occupancy and why the block is 128

This kernel is **register- and math-heavy**: each thread holds the Newton state and
runs several `exp`/`log`-laden passes. A moderate block of **128 threads** (4 warps)
keeps enough warps resident to hide the latency of the transcendental units and the
occasional constant-cache miss, without so many threads that register pressure
forces spills on sm_75. Because the kernel is compute-bound (little global traffic),
throughput is limited by the SM's math pipes, not memory bandwidth — the opposite of
a stencil like flagship 6.04. See the block-size sweep exercise in the README.

### Where the catalog's cuFFT / cuBLAS would go

The catalog mentions cuFFT (spectral filter) and cuBLAS (joint iterative
reconstruction). Those belong to the **fuller image-domain / ADMM** pipeline, not
this focused projection-domain solve: cuFFT would implement the ramp-filter step of
a filtered backprojection that turns each material sinogram into an image (see
project 4.01), and cuBLAS would drive the linear-algebra of an ADMM reconstruction
that couples the energy channels. This project stops at the per-bin decomposition —
the didactic core — and describes the rest here.

---

## Numerical considerations

- **Double precision throughout.** Material decomposition is **ill-conditioned**:
  the two basis materials' attenuation curves are only *modestly* different, so the
  Jacobian `det = J00·J11 − J01·J10` can be small and small measurement errors are
  amplified. FP64 keeps the 2×2 inverse and the accumulation of `exp` terms clean.
  The FP32-vs-FP64 exercise makes this ill-conditioning tangible.
- **`det` guard.** If the two spectra "see" the material almost identically (small
  `det`), the inverse blows up. We floor `|det|` at `1e-12` so the step stays
  finite; in a production solver you would instead switch to a damped
  (Levenberg–Marquardt) step or reject the bin.
- **Transmission guard.** `T` (the weighted transmission) is floored at `1e-300`
  before the `log`/division so a fully-absorbed ray cannot produce `inf`/`NaN`.
- **Non-negativity clamp.** Path lengths are physical (`≥ 0`); clamping after each
  step keeps Newton on the physical branch and mirrors real DECT solvers.
- **Convergence.** Newton is locally **quadratic**; from the linear seed it reaches
  `NEWTON_TOL = 1e-12` in ~3–5 iterations here (the demo prints the count per bin).
- **Determinism.** Every bin's arithmetic is independent and there are **no
  atomics**, so the result does not depend on thread scheduling — stdout is
  byte-identical every run (timings go to stderr). This is the determinism rule of
  `docs/PATTERNS.md §3`, achieved here trivially because nothing is reduced across
  threads.

---

## How we verify correctness

Two independent checks:

1. **GPU == CPU to machine precision.** The per-bin physics (forward model,
   Jacobian, Newton step, linear seed) lives in **one `__host__ __device__` header**
   (`src/dect.h`). The CPU reference loops it; the GPU kernel calls it from one
   thread. Same operations, same order, same FP64 → the results agree to the last
   bit. The demo asserts `max|GPU − CPU| ≤ 1e-9 cm`; the *actual* difference is
   ~`7e-15 cm` (rounding noise only). This is the "same exact operations on both
   sides → exact tolerance" case of `docs/PATTERNS.md §4`.

2. **Recovery vs a known answer (the science check).** The synthetic sample is
   generated by `make_synthetic.py` from **known** `(t1, t2)` truths using the
   *same* forward model (5) the code inverts. The demo reports `max recovery error
   vs known truth ≈ 6.8e-6 cm`. That residual is **not** solver error — it is the
   6-decimal rounding of the stored measurements, amplified modestly by the
   Jacobian conditioning. It validates that the whole forward-then-inverse loop is
   self-consistent (PATTERNS.md §6, "embed a known answer and recover it").

Edge cases exercised by the sample: bins with **zero iodine** (`t2 ≈ 0`, no
contrast) and a wide range of body-size soft-tissue paths (`t1` from 2 to 30 cm).

---

## Where this sits in the real world

This is a deliberately **reduced-scope teaching version** (CLAUDE.md §13). Real
spectral CT differs in several ways:

- **Real spectra and cross-sections.** Production code uses measured or simulated
  tube spectra (e.g. **SpekPy**) and **NIST XCOM** attenuation cross-sections, not
  the smooth analytic bumps and `1/E³ + const` curves here. The *structure* of the
  math is identical; only the tables change.
- **Calibration instead of first-principles.** Many clinical scanners fit a
  **polynomial** `t = P(m_lo, m_hi)` to phantom calibration data rather than solving
  the physical forward model — faster, and it absorbs unmodeled effects (scatter,
  detector response). Our Newton solver is the physics-based reference such
  calibrations approximate.
- **Photon-counting CT (PCCT).** New detectors resolve **4–8 energy bins**, turning
  the 2×2 solve into a larger (over-determined) nonlinear least-squares — Gauss-
  Newton per bin — and enabling **K-edge imaging**: a contrast agent (iodine,
  gadolinium) has a sharp jump in μ at its K-edge, so binning across that edge lets
  you image *that specific element*. That is the exercise-1 extension.
- **Image-domain and iterative methods.** Instead of decomposing the sinogram, one
  can reconstruct each energy and decompose in image space, or solve a coupled
  **ADMM** problem that regularizes across energies (the arXiv:1905.00934 splitting
  method the catalog cites). Those add cuFFT (filtering) and cuBLAS (linear algebra)
  to the pipeline — described in "GPU mapping" above.
- **Production toolkits.** **ASTRA**, **TIGRE**, and **ODL** provide the
  projection/backprojection and operator machinery you would combine with this
  decomposition step to build a full spectral-CT reconstruction.

The one thing that does **not** change from toy to clinic: it is a huge number of
**independent per-bin nonlinear solves**, which is why the GPU mapping here is the
same one the real systems use.
