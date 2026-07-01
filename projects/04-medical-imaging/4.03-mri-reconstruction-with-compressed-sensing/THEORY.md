# THEORY — 4.3 MRI Reconstruction with Compressed Sensing

> The deep didactic explanation (the "why"). Written for a sharp student who knows
> C++ but is new to CUDA and new to this domain. See [README.md](README.md) for the
> quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

### What an MRI scanner actually measures

A magnetic-resonance scanner never captures a picture directly. It excites hydrogen
nuclei (mostly in water and fat) with radio-frequency pulses in a strong magnetic
field, then applies **gradient fields** that make the precession frequency and phase
vary with position. The signal the receive coil picks up at a given moment is a
*single complex number*: it is the integral of the whole slice, weighted by a spatial
sinusoid whose frequency is set by the gradients. Formally, that number is one sample
of the image's **2D Fourier transform**. The space of all such samples is called
**k-space**.

To form an image the scanner must sweep the gradients to visit many k-space
locations — traditionally, one horizontal line of k-space per "shot," with a brief
wait between shots for the spins to relax. A full 256-line scan therefore takes on
the order of minutes. Long scans mean patient discomfort, motion blur, and expensive
scanner time.

### Why compressed sensing helps

Here is the key clinical observation: **medical images are compressible.** An MR
angiogram is mostly black with a few bright vessels; a brain image is piecewise
smooth. "Compressible" means there is some transform (wavelets, spatial gradients)
in which the image has very few large coefficients. Compressed sensing (Candès,
Romberg, Tao; Donoho; and Lustig's application to MRI, 2007) proves that if a signal
is sparse in a known transform, you can reconstruct it from **far fewer measurements
than Nyquist demands** — provided the samples are taken *incoherently* (here: a
random, variable-density subset of k-space lines). So we deliberately **skip 60–75%
of the k-space lines**, cutting scan time 3–4×, and recover the missing data by
solving an optimization problem that says "find the image consistent with the samples
we did take that is also sparse."

This project builds that solver end-to-end for a single slice.

---

## 2. The math

### Notation

| Symbol | Meaning |
|---|---|
| `x ∈ ℂ^{n×n}` | the image we want (complex; magnitude `|x|` is what is viewed) |
| `F` | the 2D discrete Fourier transform operator (unitary up to a scale) |
| `M` | the sampling mask: a diagonal 0/1 operator keeping acquired k-space entries |
| `y = M F x_true + noise` | the measured, under-sampled k-space (0 where unsampled) |
| `E = M F` | the forward/encoding operator (image → measured k-space) |
| `Ψ` | a sparsifying transform (here `Ψ = I`, the identity; TV/wavelet in production) |
| `λ ≥ 0` | regularization weight trading data-fit against sparsity |

### The optimization problem

Compressed-sensing reconstruction solves the convex program

```
  minimize_x   (1/2) ‖ M F x − y ‖₂²   +   λ ‖ Ψ x ‖₁
               \___________________/       \_________/
                 data consistency           sparsity prior
```

The first term pulls `x` toward agreeing with the samples we measured. The second,
the **L1 norm**, is the sparsity-promoting magic: minimizing an L1 norm (unlike L2)
drives small coefficients to *exactly zero*, so the solution is sparse — which is the
prior knowledge that lets us fill in the un-measured k-space.

For a real-valued sparsity penalty on complex data we apply the prox to the real and
imaginary parts independently (the standard toolbox choice); `cs_core.h`'s
`soft_threshold_cplx` does exactly this.

---

## 3. The algorithm

### FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)

The objective is convex but non-smooth (the L1 term has a kink at 0), so we use a
**proximal-gradient** method: take a gradient step on the smooth data term, then apply
the *proximal operator* of the non-smooth term. The prox of `λ‖·‖₁` is the
**soft-threshold** (shrinkage): `prox(v) = sign(v)·max(|v|−λ, 0)`.

The gradient of the data term `½‖M F x − y‖²` is

```
  ∇(x) = Fᴴ M (M F x − y)  =  F⁻¹{ M (F x − y) }
```

because `F` is (up to scale) unitary and `M` is a 0/1 diagonal (`MᴴM = M`). So one
gradient evaluation costs **one forward FFT + one inverse FFT**.

Plain proximal-gradient (ISTA) converges at `O(1/k)`. **FISTA** adds Nesterov
momentum — extrapolating each new iterate past the previous one — to get `O(1/k²)`
with no extra FFTs. The full loop (implemented identically in `reference_cpu.cpp` and
`kernels.cu`):

```
  x₀ = z₀ = F⁻¹{y}            # warm start: the zero-filled adjoint image
  θ₀ = 1
  for k = 0,1,2,...:
      g  = F⁻¹{ M (F zₖ − y) }             # gradient  (2 FFTs)
      xₖ₊₁ = softThreshold(zₖ − t·g, t·λ)   # prox-gradient step   (t = 1)
      θₖ₊₁ = (1 + √(1 + 4θₖ²)) / 2          # momentum bookkeeping
      β    = (θₖ − 1) / θₖ₊₁
      zₖ₊₁ = xₖ₊₁ + β (xₖ₊₁ − xₖ)           # Nesterov extrapolation
  return |x|
```

The **step size `t = 1`** is valid because the Lipschitz constant of the data term's
gradient equals the squared spectral norm of `E = M F`, and with our unit-scaled FFT
convention `‖E‖ ≤ 1`, so `L = 1` and `t = 1/L = 1`.

### Complexity

Per iteration: two `n×n` FFTs at `O(n² log n)` each, plus three `O(n²)` per-pixel
passes. For `K` iterations the total is `O(K n² log n)` — the FFTs dominate. Serial
(CPU) and parallel (GPU) do the *same* arithmetic; the GPU parallelizes each FFT
across all `n²` points and each per-pixel pass across all pixels. The data-access
pattern is FFT-internal (strided, handled by cuFFT) plus fully coalesced streaming
reads/writes in the per-pixel kernels — high effective bandwidth, low arithmetic
intensity (the classic FFT-bound regime).

---

## 4. The GPU mapping

### Who does what

| Step | Runs on | Why |
|---|---|---|
| forward FFT `F z` | **cuFFT** `cufftExecC2C(…, CUFFT_FORWARD)` | solved, bandwidth-bound; library is optimal |
| mask residual `M(Fz − y)` | custom kernel, 1 thread/pixel | trivial elementwise, fused with the shared formula |
| inverse FFT `F⁻¹{·}` | **cuFFT** `cufftExecC2C(…, CUFFT_INVERSE)` | same as forward |
| scale by `1/n²` | custom kernel | cuFFT leaves the inverse un-normalized |
| prox-gradient `soft(z−t·g)` | custom kernel, 1 thread/pixel | the CS "shrinkage" step |
| momentum `x+β(x−x_prev)` | custom kernel, 1 thread/pixel | cheap extrapolation |

### Thread-to-data mapping (the per-pixel kernels)

All custom kernels use the most basic CUDA mapping — **one thread per pixel** over the
flat `n²` image:

```
  grid  = ceil(n² / 256) blocks      block = 256 threads
  pixel index  i = blockIdx.x * blockDim.x + threadIdx.x     (guard i < n²)
```

256 threads/block is a solid default on sm_75–sm_89 (multiple of the 32-lane warp,
enough warps to hide latency, many resident blocks for occupancy). These kernels are
memory-bound elementwise streams; there is no data reuse, so **no shared memory or
atomics are needed** — each pixel updates independently. (Contrast a NUFFT gridding
kernel, which *would* need shared-memory accumulation and atomics; see §7.)

```
   image x  (n×n, row-major)              cuFFT plan (n×n C2C)
   ┌───────────────┐                      ┌─────────────────────┐
   │ pixel 0 1 2 … │  ── F (forward) ──▶  │  X = Σ x·e^{-2πi…}   │
   │  …            │                      └─────────┬───────────┘
   │  … n²−1       │                                │  mask kernel (1 thr/pixel)
   └───────────────┘                       r = M (X − y)   [in place]
        ▲                                           │
        │  prox-gradient kernel (1 thr/pixel)       ▼
        └──────────  g = F⁻¹{r}·(1/n²)  ◀── F⁻¹ (inverse) ── cuFFT
```

### cuFFT is not a black box (CLAUDE.md §6.1.6)

`cufftPlan2d(&plan, n, n, CUFFT_C2C)` builds a plan for one `n×n` complex-to-complex
2D FFT laid out **row-major** (row stride `n`) — exactly our `Cplx` buffer layout.
`cufftExecC2C(plan, in, out, CUFFT_FORWARD)` computes
`X[k₁,k₂] = Σ_{r,c} x[r,c] · exp(−2πi(k₁r + k₂c)/n)`, the same double sum
`fft2_cpu` computes by hand via a separable radix-2 Cooley-Tukey FFT (rows then
columns). `CUFFT_INVERSE` uses `+i` and is **un-normalized**, so we multiply by
`1/n²` ourselves — precisely what `ifft2_cpu` does. Hand-rolling this for the GPU
would mean writing and tuning the bit-reversal + butterfly across both dimensions;
cuFFT does it faster and correctly, which is the whole point of using the library.

`Cplx` is `{float re; float im;}`, bit-identical to cuFFT's `cufftComplex` (`float2`),
so we `reinterpret_cast` between them with zero copying — guarded by a
`static_assert` in `kernels.cu`.

### Reusing the plan and staying on-device

The plan is built **once** and reused for all `2K` transforms. The entire FISTA loop
runs on the device with **no per-iteration host↔device copies** — image estimates
live in device buffers and only the final magnitude image is copied back. That is the
production discipline: keep the working set resident and let cuFFT + tiny kernels
chew on it.

---

## 5. Numerical considerations

- **Precision.** We use **FP32**, matching real scanners and cuFFT's fast path.
  The reference FFT accumulates its butterflies in **double** internally (for a clean
  baseline) but stores FP32 results, so both paths carry FP32 data through FISTA.
- **The soft-threshold is deterministic and branch-consistent.** `soft_threshold_real`
  in `cs_core.h` avoids `std::fabs`/`std::copysign` (which have host-only overloads
  that warn under nvcc) in favor of explicit sign branches, so host and device emit
  identical instructions and identical bits. `c_abs` uses `sqrtf` (host+device) for
  the same reason.
- **No atomics, no reduction reordering.** Every kernel is a pure elementwise map —
  no `atomicAdd`, no parallel sum — so there is **no floating-point non-associativity**
  to worry about (PATTERNS.md §3 rule 2). The only source of CPU/GPU divergence is the
  FFT library itself (cuFFT vs our radix-2), which differ by a few ULP per transform.
- **Determinism of stdout.** Every number printed to stdout is computed from the
  **deterministic CPU path**, so the demo output is byte-identical every run
  regardless of GPU scheduling. Timings and the (tiny, run-varying) GPU/CPU error go
  to stderr. Verified: 3 consecutive runs produce identical stdout.
- **Convergence, not exactness.** FISTA is an iterative minimizer; the "answer" is the
  fixed point it approaches. With `λ` and iteration count fixed in the sample, both
  paths take the *same* deterministic trajectory, so they land on the same image.

---

## 6. How we verify correctness

Two independent checks, both in `main.cu`:

1. **GPU vs CPU (portability / correctness).** The GPU (cuFFT) and CPU (hand radix-2
   FFT) run the *identical* FISTA arithmetic — every per-pixel formula comes from the
   one shared `cs_core.h` (the `__host__ __device__` idiom, PATTERNS.md §2) — so the
   only difference is the FFT engine. We compare the final magnitude images by RMS
   difference and require it below `tol = 2e-3 + 1e-3·peak`. This is the **iterative-
   solver tolerance** of PATTERNS.md §4: over 60 iterations the two FFT libraries'
   rounding diverges by a small, physically-negligible amount, so we verify to a
   physical tolerance rather than pretending bit-identity. *Observed* agreement is far
   tighter — RMS ≈ `3e-8` — because both FFTs are accurate and the math is shared.

2. **CS vs zero-filling (the science).** Because the sample is synthetic, we know the
   ground-truth image. We check that FISTA's error against the truth is **smaller than
   the zero-filled baseline's** error. On the committed sample CS is ~8.8× better —
   quantitative proof that the sparsity prior recovered information the naive inverse
   FFT could not.

Edge cases handled: the loader rejects a non-power-of-two `n` (the radix-2 FFT
requires it) and a truncated file; unsampled k-space entries contribute zero residual
(`data_consistency_residual`), so the mask is honored exactly.

---

## 7. Where this sits in the real world

This is a deliberately **reduced-scope teaching version** (CLAUDE.md §13). Production
CS-MRI, as in **BART**, **SigPy**, **MIRT**, and **PyNUFFT**, differs in scale and
sophistication:

- **Non-Cartesian trajectories (NUFFT).** Real fast scans use radial or spiral
  k-space paths, whose samples do not fall on the FFT grid. Production code uses a
  **NUFFT**: convolve each off-grid sample onto a grid with a Kaiser-Bessel kernel
  (this *is* the gridding step that needs shared-memory accumulation and atomics),
  apply a plain FFT, then deapodize. We use pure Cartesian sampling so a plain FFT
  suffices — the iteration structure is otherwise identical.
- **Parallel imaging (SENSE / GRAPPA / PICS).** ~32 receive coils each see the image
  through a spatial **sensitivity map** `S_c`. The data term becomes
  `Σ_c ‖M F S_c x − y_c‖²`; cuFFT **batches** the per-coil FFTs in a single call and
  cuBLAS combines coils. This multiplies compute by the coil count — the main reason
  GPUs are essential in practice.
- **Better sparsity & solvers.** Real priors are **wavelet** and **total-variation**
  (TV), often several combined; solvers include **ADMM / Split-Bregman** (which split
  the problem so each sub-step has a closed form) and learned/unrolled networks. FISTA
  with an identity prior is the clean pedagogical entry point; swapping `Ψ` and the
  solver into this same loop is Exercise 3.
- **3D + dynamic (k-t).** Clinical volumes are 3D (`~256³`) and dynamic sequences add
  time (k-t SENSE), where sparsity in the temporal Fourier domain gives huge
  acceleration.

The through-line: every one of these is *this loop* — forward encode, data-consistency
residual, adjoint, proximal shrinkage, momentum — with a richer encode operator `E`
and a richer prox. Understanding the single-coil Cartesian case makes the rest
readable.

---

## References

- M. Lustig, D. Donoho, J. Pauly, "Sparse MRI: The application of compressed sensing
  for rapid MR imaging," *Magn. Reson. Med.*, 2007 — the founding CS-MRI paper.
- A. Beck, M. Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear
  Inverse Problems," *SIAM J. Imaging Sci.*, 2009 — FISTA.
- E. Candès, J. Romberg, T. Tao / D. Donoho (2006) — the compressed-sensing theory.
- **BART** <https://github.com/mrirecon/bart> — study `pics` and its operator/prox
  abstractions to see this loop at production scale.
- **SigPy** <https://github.com/mikgroup/sigpy> — readable Python/CuPy linear
  operators and app classes for MRI.
- **MIRT** <https://github.com/JeffFessler/MIRT.jl> — deep on the reconstruction math.
- **PyNUFFT** <https://github.com/jyhmiinlin/pynufft> — the NUFFT gridding this
  project simplifies away.
- NVIDIA **cuFFT** documentation — plan/exec semantics, layouts, and batching.
