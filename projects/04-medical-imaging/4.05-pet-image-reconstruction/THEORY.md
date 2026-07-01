# THEORY — 4.5 PET Image Reconstruction

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

**Positron Emission Tomography (PET)** images *function*, not anatomy. A molecule
of interest (a tracer, e.g. the glucose analogue ¹⁸F-FDG) is labelled with a
**positron-emitting** isotope and injected. Wherever the tracer accumulates —
tumours light up because they burn glucose fast — the isotope decays and emits a
**positron**. The positron travels a fraction of a millimetre, then **annihilates**
with an electron, converting their mass into **two 511 keV gamma photons flying in
almost exactly opposite directions** (conservation of momentum).

A ring of scintillation detectors surrounds the patient. When two detectors fire
within a few nanoseconds, the scanner records a **coincidence**: the annihilation
happened somewhere on the straight line joining those two detectors — a **Line Of
Response (LOR)**. It does *not* know *where* on the line. Over a scan, millions of
coincidences accumulate into a count per LOR. Sorted by the LOR's angle and offset,
those counts form a **sinogram** (a point source traces a sine curve through it —
hence the name).

**The reconstruction problem:** given the sinogram of counts `y`, recover the 3-D
(here 2-D) map of tracer concentration `x` that produced it. This is a tomographic
inverse problem, like CT (project 4.01) — but with a crucial twist: PET counts are
*few* and *random*. Each count is a Poisson event. That statistics changes the
right way to invert, from CT's linear filtered backprojection to PET's **iterative,
statistically-principled MLEM**.

## 2. The math

**Discretize.** Let the image be `x ∈ ℝ^{P}` (P = N² pixels, `x_j ≥ 0` = activity in
pixel j) and the data `y ∈ ℝ^{M}` (M = K·D LORs, `y_i` = counts on LOR i). The
scanner is a linear **system matrix** `A ∈ ℝ^{M×P}`:

```
A_ij = probability that a decay in pixel j is detected on LOR i.
```

The **forward model** is that the expected counts on LOR i are `(A x)_i = Σ_j A_ij x_j`,
and the actual counts are Poisson around that mean:

```
y_i ~ Poisson( (A x)_i ).
```

**Maximum likelihood.** We seek the image that makes the observed counts most
probable. The Poisson log-likelihood (dropping constants) is

```
L(x) = Σ_i [ y_i · log( (A x)_i )  −  (A x)_i ].
```

Maximizing `L` over `x ≥ 0` has no closed form, but the **Expectation-Maximization**
recipe yields the beautifully simple multiplicative **MLEM update** (Shepp & Vardi
1982):

```
x_j^{n+1} = ( x_j^n / s_j ) · Σ_i A_ij · ( y_i / (A x^n)_i ),      s_j = Σ_i A_ij.
```

Read it as four operations: **forward project** `A x^n`, form the **ratio**
`y / (A x^n)`, **back-project** that ratio `Aᵀ(·)`, and **rescale** by the current
image over the **sensitivity** `s = Aᵀ1`. MLEM has three lovely properties: it keeps
`x ≥ 0` automatically (all factors are non-negative), it never decreases the
likelihood, and it approximately conserves total counts.

**Our system matrix.** We use the same 2-D **parallel-beam** geometry as the CT
flagship, defined once in [`src/pet_geometry.h`](src/pet_geometry.h):

- Angles `θ_k = k·π/K`, `k = 0…K−1` (a parallel-beam rebinning over 180°).
- Detector bin j at signed offset `s_j = (j − (D−1)/2)·ds`.
- A pixel at world `(wx, wy)` projects, at angle k, to fractional bin
  `fidx = (wx·cosθ_k + wy·sinθ_k)/ds + (D−1)/2`. Its activity is split **linearly**
  between bins `⌊fidx⌋` and `⌊fidx⌋+1` with weights `(1−w)` and `w`, `w = fidx − ⌊fidx⌋`.

That split *defines* the non-zero entries of `A`. Because forward and back
projection use the **same** split, back-projection is the exact transpose `Aᵀ` — a
requirement for MLEM to behave.

## 3. The algorithm

```
sensitivity:  s = Aᵀ 1                              # once, before iterating
init:         x = 1                                  # uniform positive image
repeat iters times:
    ŷ = A x                                          # forward projection  (M sums)
    r_i = y_i / ŷ_i     (0 if ŷ_i = 0)              # ratio                (M divides)
    c = Aᵀ r                                          # back projection     (P sums)
    x_j = x_j · c_j / s_j   (freeze if s_j = 0)      # multiplicative update (P ops)
```

**Complexity.** With our pixel-driven projector each pixel touches all K angles, so
one forward *or* back projection costs `O(N²·K)`, and the whole reconstruction is
`O(iters · N²·K)`. A Siddon ray tracer would cut the per-LOR work to `O(N)` pixels
on the ray instead of `O(N²)`, giving `O(iters · K·D·N)` — the production choice.

**Data-access pattern.** Both projections are **gathers**: each output element
(an LOR count, or a pixel update) reads many inputs and writes one value. That is
the same shape as CT backprojection (docs/PATTERNS.md, the 4.01 pattern), which is
exactly why it parallelizes without atomics.

## 4. The GPU mapping

One MLEM iteration is three kernel launches (see [`src/kernels.cu`](src/kernels.cu)):

**(a) `forward_project_kernel` — one thread per LOR.**
Linear thread index `i` maps to LOR `(k = i/D, j = i%D)`. The thread sweeps every
pixel and gathers those whose linear split lands in *its* bin j at angle k, into a
`double` accumulator, and writes `ŷ[i]`. One output per thread → **no atomics**.
Launch: 1-D grid, `ceil(K·D / 256)` blocks × 256 threads.

**(b) `ratio_kernel` — one thread per LOR.** Element-wise `r = y/ŷ` (SAXPY-shaped,
guarded divide).

**(c) `update_kernel` — one thread per pixel** on a 2-D tile grid (16×16), the same
mapping CT backprojection uses. Thread `(px,py)` gathers the back-projected ratio
over all K angles (same split weights → the transpose `Aᵀ`) and applies the
multiplicative update `x ← x·c/s` **in place**. Independent pixel outputs → **no
atomics**.

```
   forward: thread per LOR (k,j)          update: thread per pixel (px,py)
   ┌─────────── K·D grid ───────────┐     ┌────────── N×N tile grid ─────────┐
   i ─▶ (k,j)                              (px,py)
   for all pixels: if split hits j         for all angles k:
       acc += x_pixel · weight                 acc += ratio[k,·] · weight
   ŷ[k,j] = acc                            x[px,py] *= acc / s[px,py]
   └────────────────────────────────┘     └───────────────────────────────────┘
        (gather, no atomics)                       (gather, no atomics)
```

**Memory hierarchy.** The image, sinogram, sensitivity, and trig tables live in
**global memory**; the cos/sin tables (length K) are small and hot, so the L1/L2
cache serves them well (a `__constant__` table is a natural upgrade — an exercise).
Each thread's accumulator is a **register** `double`. **All reconstruction state
stays on the device across iterations** — only the final image is copied back — so
per-iteration cost is pure compute (the whole point of GPU MLEM). We time the loop
with CUDA events (transfers happen once, outside it).

**Why not the "obvious" LOR-parallel backprojection?** The textbook GPU
backprojection assigns one thread per LOR and **scatters** its contribution into the
image with `atomicAdd`. That works, but a *float* atomic sum depends on the
(nondeterministic) order threads arrive → the stdout would vary run to run
(docs/PATTERNS.md §3). We deliberately use the **pixel-parallel gather** instead:
every reduction happens inside one thread, in a fixed order → deterministic and
exactly transpose-consistent with the forward pass.

**Where a library would help (no black boxes).** The catalog mentions cuBLAS for
correction factors and warp-level reductions for scatter. In a full pipeline,
per-LOR normalization/attenuation is an element-wise vector multiply (`cublasSdgmm`
or a trivial kernel), and scatter/randoms estimation involves reductions best done
with CUB/`__shfl_down_sync` warp primitives. We hand-roll the (few) reductions here
so the arithmetic is visible; the exercises point at the library versions.

## 5. Numerical considerations

- **Precision.** Image/sinogram state is **FP32** (clinical PET is fine in single
  precision — the data is noisy). But every ray/pixel **sum** accumulates in a
  **`double`** register, so the projection is well-conditioned regardless of how
  many terms it adds. Both CPU and GPU do this identically.
- **CPU/GPU parity.** The per-element geometry is a single `__host__ __device__`
  header ([`pet_geometry.h`](src/pet_geometry.h)), and the trig tables are computed
  once on the host and *uploaded* (so device `cosf` never disagrees with host
  `std::cos` in the last bit). The remaining difference is only the **order** of
  summation: the CPU forward pass scatters pixel→bins, the GPU forward pass gathers
  bins←pixels. Floating-point addition is not associative, so these differ by
  rounding — see §6.
- **Guards.** `y/ŷ` and `x·c/s` both guard division by zero (an LOR with no modeled
  counts contributes 0; a pixel no LOR sees is frozen). Counts are clamped ≥ 0 at
  load. MLEM's non-negativity is automatic.
- **Determinism.** Both kernels are **gathers with per-thread reductions**, so the
  stdout is byte-identical every run (verified). No atomics anywhere.

## 6. How we verify correctness

Two independent implementations of the *same* math — the serial CPU reference
([`reference_cpu.cpp`](src/reference_cpu.cpp)) and the parallel GPU kernels — run
the full 30-iteration MLEM and we compare the final images with
`util::max_abs_err`. Agreement between an obviously-correct serial loop and a
very-different parallel gather is strong evidence both are right.

**Tolerance = `1e-3` (absolute).** The two sides sum the same terms in a different
order; that rounding difference, compounded multiplicatively over 30 iterations,
lands around `6e-5` on the sample — comfortably under `1e-3`. This is the same
honest "long iterative solver, FMA-level divergence" tolerance used by 4.01 and
10.02 (docs/PATTERNS.md §4). We do **not** claim bit-exactness.

**A second, physical check.** Because the sample sinogram was forward-projected
from a *known* phantom (a big central disc + a small off-center hot spot), the
reconstruction is checked against reality, not just against itself: `main.cu`
reports that the center pixel is bright, the **peak** lands on the hot spot at
`(21,20)`, and the central-row profile is high in the middle and near zero at the
edges — i.e. MLEM recovered the object we put in.

## 7. Where this sits in the real world

Production PET reconstruction (STIR, SIRF, parallelproj, CASToR) keeps this exact
MLEM skeleton but adds, roughly in order of importance:

- **OS-EM** (Hudson–Larkin): update after each of S angle *subsets*, giving ~S×
  faster convergence — the clinical default. (A one-line change here; exercise 1.)
- **A real 3-D projector** — Siddon/Joseph ray tracing through a cylindrical ring
  with oblique LORs and time-of-flight (TOF) kernels that localize the annihilation
  along the LOR, sharpening the image.
- **Physics corrections** folded into `A`: **attenuation** (from a CT/MR μ-map),
  **scatter** (model-based estimation), **randoms** (delayed-coincidence subtraction),
  detector **normalization**, and **PSF/resolution modelling**.
- **Regularization** — MAP-EM with a Gibbs/quadratic prior, or MR-guided priors in
  joint **PET/MRI** reconstruction (SIRF's specialty), to control the noise MLEM
  amplifies at high iteration counts.
- **List-mode ML-EM** — iterate over the ~10⁸ individual events instead of a binned
  sinogram, exact for sparse/TOF data.
- **Dynamic PET** — reconstruct a *time series* of frames for kinetic modelling,
  multiplying the cost by the number of frames and making GPU acceleration
  essential.

Our version keeps the statistically-correct core (Poisson MLEM, matched projector
pair, sensitivity normalization) so the *idea* is honest, and omits the geometry
and corrections that make a scanner clinical.

---

## References

- **H. M. Hudson & R. S. Larkin (1994)**, "Accelerated image reconstruction using
  ordered subsets of projection data", *IEEE TMI* — the OS-EM accelerator.
- **L. A. Shepp & Y. Vardi (1982)**, "Maximum likelihood reconstruction for emission
  tomography", *IEEE TMI* — the original MLEM derivation.
- **STIR** (<https://github.com/SyneRBI/STIR>) — read `ProjectorByBin` and the OSMAPOSL
  reconstructor to see a production system-matrix + iteration API.
- **parallelproj** (<https://github.com/gschramm/parallelproj>) — a clean, fast
  CUDA/OpenCL Joseph projector; the model for exercise 3 (Siddon/Joseph).
- **SIRF** (<https://github.com/SyneRBI/SIRF>) + **SIRF-Exercises** — joint PET/MR and
  the source of openly usable phantom data.
- **A. J. Reader et al.**, reviews of list-mode and 4-D PET reconstruction — for the
  TOF/list-mode/dynamic extensions in §7.
