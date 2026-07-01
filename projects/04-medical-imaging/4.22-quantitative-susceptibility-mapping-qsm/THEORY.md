# 4.22 — Quantitative Susceptibility Mapping (QSM): the theory

> Read this alongside the code. The catalog deep-dive and `README.md` give the
> short version; this file is the long one, ordered science → math → algorithm →
> GPU mapping → numerics → verification → real-world.

---

## The science

An MRI scanner works because tissue sits in a strong, uniform main field **B₀**
(here along +z). Every tissue has a magnetic **susceptibility** χ — a
dimensionless number saying how strongly it magnetizes in that field:

- **Paramagnetic** (χ > 0): iron-rich deep-brain nuclei (globus pallidus, red
  nucleus, substantia nigra), deoxygenated venous blood, microbleeds.
- **Diamagnetic** (χ < 0): calcifications, myelin.

Those tiny susceptibilities perturb the local magnetic field. A gradient-echo
(GRE) sequence records **phase** proportional to the accumulated field shift, so
after a few standard preprocessing steps (phase unwrapping, then background-field
removal) you are left with a **local tissue field map** δB(**r**). **QSM** is the
inverse problem: recover the susceptibility map χ(**r**) that produced δB.

Why bother? χ is a *quantitative, reproducible* tissue property, unlike raw phase
(which depends on echo time, orientation, and shim). QSM is used to measure brain
iron in ageing and neurodegeneration, to map venous oxygenation, to distinguish
calcification (diamagnetic) from haemorrhage (paramagnetic) — two things that look
identical on many other sequences but have *opposite-sign* χ.

The full clinical pipeline is: **phase unwrapping** (PUROR, ROMEO) → **background
field removal** (PDF, SHARP, V-SHARP) → **dipole inversion** (this project). We
implement the last, computationally hardest stage and assume the first two are
done (our synthetic input is already a clean local field map).

---

## The math

### The forward model (dipole convolution)

A single point of susceptibility acts as a magnetic **dipole**, producing the
classic bipolar "bowtie" field around it. Because Maxwell's equations are linear,
the total field shift is the susceptibility distribution **convolved** with the
unit dipole response d(**r**):

$$\delta B(\mathbf r) = \big(d * \chi\big)(\mathbf r).$$

Convolution in space is a **pointwise multiply in k-space** (the convolution
theorem). The dipole's Fourier transform is analytically known — the **dipole
kernel**:

$$D(\mathbf k) = \frac{1}{3} - \frac{(\mathbf k \cdot \hat{\mathbf B}_0)^2}{|\mathbf k|^2}.$$

With **B₀ ∥ z**, $\hat{\mathbf B}_0=(0,0,1)$ and $\mathbf k\cdot\hat{\mathbf B}_0=k_z$, so

$$\boxed{\,D(\mathbf k) = \frac{1}{3} - \frac{k_z^2}{k_x^2+k_y^2+k_z^2}\,}$$

and the forward model is simply

$$\widehat{\delta B}(\mathbf k) = D(\mathbf k)\,\hat\chi(\mathbf k).$$

This exact formula is `dipole_kernel()` in [`src/qsm_core.h`](src/qsm_core.h).

### Why the inverse is ill-posed (the magic angle)

To recover χ you divide by D(**k**): $\hat\chi = \widehat{\delta B}/D$. But

$$D(\mathbf k)=0 \iff k_z^2=\tfrac{1}{3}|\mathbf k|^2 \iff \theta=\arccos\!\sqrt{\tfrac13}\approx 54.7^\circ,$$

a **double cone** at the *magic angle* to B₀. On that cone the forward model
multiplies χ by zero — it **erases** all information there — so the naive inverse
$1/D$ is infinite and amplifies noise into severe **streaking artifacts**. Every
QSM method is a strategy for the bins where D ≈ 0.

### Method 1 — TKD (Threshold-based K-space Division)

Clamp the *magnitude* of D away from zero before inverting (Shmueli 2009, Wharton
2010):

$$w_{\text{TKD}}(\mathbf k)=\frac{1}{\operatorname{sign}(D)\,\max(|D|,\,t)},\qquad
\hat\chi = w_{\text{TKD}}\cdot\widehat{\delta B}.$$

Where |D| is large the weight is the faithful 1/D; near the cone it is capped at
±1/t (bounded). Threshold t (≈ 0.1–0.2) is a **bias/variance knob**: larger t
kills streaking but *underestimates* χ. This is `tkd_reciprocal()`.

### Method 2 — Tikhonov-regularized least squares

Pose inversion as regularized least squares:

$$\min_{\hat\chi}\ \big\|\,D\odot\hat\chi-\widehat{\delta B}\,\big\|^2
   +\alpha\,\|\hat\chi\|^2 .$$

Because D is **diagonal in k-space**, this **decouples bin by bin**. Setting the
gradient to zero gives a closed-form Wiener filter:

$$\hat\chi(\mathbf k)=\frac{D}{D^2+\alpha}\,\widehat{\delta B}(\mathbf k)
   \quad\Rightarrow\quad w_{\text{Tik}}=\frac{D}{D^2+\alpha}.$$

As α→0 this tends to 1/D (ill-posed); a positive α bounds it near the cone (there
D≈0 so the weight ≈ D/α → 0, gently zeroing unreliable bins). This is
`tikhonov_exact_weight()`.

---

## The algorithm

Three reconstructions share one skeleton — **FFT → per-bin weighting → inverse
FFT** — differing only in the weight:

```
load field map δB
────────────────────────── direct methods (one shot)
Fδ = FFT3(δB)
for each k-space bin:  Fχ[k] = w(k) · Fδ[k]       # w = TKD  or  Tikhonov-Wiener
χ = real( IFFT3(Fχ) ) / N
────────────────────────── iterative method (the GPU-headline pattern)
Fδ = FFT3(δB)          # once
Fχ = 0
repeat `iters` times, for each bin independently:
    r = D·Fχ − Fδ                       # residual (forward model − data)
    g = 2( D·r + α·Fχ )                 # gradient of the Tikhonov objective
    Fχ = Fχ − step·g                    # gradient-descent step
χ = real( IFFT3(Fχ) ) / N
```

The per-bin gradient step is `tikhonov_grad_step()`; the loop is
`reconstruct_tikhonov_iter_*`. It is **verified to converge** to the closed-form
Wiener minimizer (§How we verify correctness).

### Complexity: serial vs parallel

| Stage | Serial (CPU reference) | Parallel (GPU) |
|---|---|---|
| One 3-D transform | **O(N²)** direct DFT (a literal sum over all voxel pairs) | **O(N log N)** cuFFT, split across thousands of threads |
| Per-bin weight / grad step | O(N) | O(N) threads, one bin each, O(1) work per thread |
| Iterative solve | O(N² + iters·N) | O(N log N + iters·N) |

We deliberately give the CPU reference an **O(N²) direct DFT** (not an FFT): it is
transparently correct — the textbook definition, no butterfly bookkeeping — which
is exactly what a teaching baseline should be, and its O(N²) cost is the whole
motivation for the FFT. At scan scale (256³ ≈ 16.7 M voxels) O(N²) is
astronomically infeasible; cuFFT's O(N log N) is milliseconds.

---

## The GPU mapping

**Pattern:** *use cuFFT for the spectral transform, custom element-wise kernels
for everything else* (PATTERNS.md §1, the same pattern as flagship `8.03` and
sibling `4.30`). See [`src/kernels.cu`](src/kernels.cu).

- **Transforms → cuFFT.** A single in-place 3-D double-complex plan
  (`cufftPlan3d(..., CUFFT_Z2Z)`) does both forward and inverse (direction is a
  flag on `cufftExecZ2Z`). cuFFT's dimension order is slowest→fastest = (nz, ny,
  nx), matching our x-fastest storage. It is **unnormalized**: a forward+inverse
  round trip scales by N = nx·ny·nz, so we divide by N once (folded into the last
  element-wise kernel).
- **Per-bin work → one thread per k-space bin.** `weight_tkd_kernel`,
  `grad_step_kernel`, and `scale_kernel` map thread
  `i = blockIdx.x·blockDim.x + threadIdx.x` to bin `i` of the flat spectrum, with
  the usual `if (i < N)` guard on the ragged last block. Block size 256 (8 warps,
  good occupancy on sm_75…sm_89). The dipole operator is **diagonal in k-space**,
  so bins are fully independent — **no shared memory, no atomics, no
  synchronization**. Each thread reads/writes only its own one or two complex
  bins; the kernels are purely bandwidth-bound.
- **Recomputing D(k) vs storing it.** `bin_dipole()` recovers (kx,ky,kz) from the
  flat index and evaluates D(**k**) from a few flops each call, rather than
  precomputing an N-element table. On a memory-bound kernel, recompute-cheap-flops
  beats an extra global-memory read — a small but real teaching point.
- **Why the iterative solve keeps the FFT *outside* the loop.** Our Tikhonov
  objective is diagonal in k-space, so the entire gradient descent runs in the
  frequency domain: FFT the data **once**, iterate on the spectrum, inverse-FFT
  **once**. A *real* edge-regularized solver (MEDI, TV) couples neighbouring
  voxels, so its gradient needs a spatial-domain operator — forcing an FFT
  **inside every iteration** (O(100) forward+inverse transforms). That is the
  catalog's stated bottleneck; our reduced-scope version isolates the k-space part
  so the cuFFT mechanics are legible, and this paragraph is the honest bridge to
  the full problem.

### Memory layout

- Volumes are `double` (FP64), stored **x fastest, then y, then z**:
  `vox[(z·ny + y)·nx + x]`. The k-space spectrum uses the identical layout, so a
  linear bin index maps to (kx,ky,kz) the same way on host and device.
- Device buffers are `cufftDoubleComplex` (== `double2`, `.x`=real `.y`=imag),
  which is **layout-identical** to the pure-C++ `Complex` in `qsm_core.h` — that
  is what lets the same per-bin math compile for both host and device.

---

## Numerical considerations

- **FP64 throughout.** The inverse divides by dipole values that get arbitrarily
  small near the magic cone; FP32 would lose precision exactly where it hurts.
  The teaching volume is small, so double is free. Real GPU QSM often uses FP32
  for speed and leans harder on regularization to tolerate it (an exercise).
- **The threshold / α are regularization, not bugs.** They *bias* χ toward
  underestimation (visible in the demo: recovered χ < ground-truth χ). That bias
  is the price of a bounded, streak-free inverse — a genuine QSM trade-off, not a
  numerical error. TKD post-correction and iterative edge priors reduce it.
- **Determinism.** No atomics, no float reductions with race-dependent order:
  every bin is independent and every reduction (RMS) runs the same order on both
  sides. So stdout is byte-identical run to run (timings go to stderr). See
  PATTERNS.md §3.
- **cuFFT vs direct DFT round-off.** The two algorithms sum in different orders
  and use fused-multiply-add differently, so results differ by ~1e-16 per voxel
  (double precision) — far below any physical scale. We verify to `atol = 1e-6`
  and report the *actual* worst error (~1e-16) on stderr; we do **not** pretend
  the results are bit-identical.
- **Reality of the imaginary residue.** χ is real, so the inverse FFT's imaginary
  part is ~1e-16 round-off; we keep only the real part. A large imaginary part
  would signal a bug (e.g. a non-Hermitian weighting).

---

## How we verify correctness

Three independent checks, all in [`src/main.cu`](src/main.cu):

1. **GPU == CPU, TKD.** RMS voxel difference between the cuFFT TKD and the
   direct-DFT TKD. Because both call the *same* `tkd_reciprocal()` (shared
   `qsm_core.h`), the only difference is the transform algorithm → RMS ≈ 1e-16.
2. **GPU == CPU, iterative.** Same idea for the 200-step gradient descent: the
   GPU and CPU run byte-identical `tikhonov_grad_step()` per bin → RMS ≈ 1e-16.
3. **The iterative solve converged (algorithm check, not just parity).** RMS
   between the iterative result and the *closed-form* Wiener minimizer
   `tikhonov_exact_weight()`. This validates that gradient descent actually solves
   the least-squares problem, independent of CPU/GPU agreement → gap ≈ 1e-8 after
   200 iterations.

A fourth, *scientific* check: we rebuild the known synthetic phantom, report the
recovered χ at the four source voxels next to the ground truth, and compute a
**data-consistency residual** — re-apply the forward dipole model to the
reconstructed χ and compare with the input field map (small residual ⇒ the
reconstruction explains the data). This validates the *science*, not just
CPU==GPU agreement (PATTERNS.md §4).

Documented tolerances: `atol = 1e-6` for GPU==CPU (they actually agree to ~1e-16),
`5e-3` for iterative→closed-form convergence (it actually reaches ~1e-8).

---

## Where this sits in the real world

Our version is a faithful but **reduced-scope** teaching QSM. Production tools
differ in ways THEORY should name honestly:

- **The rest of the pipeline.** Real QSM must first **unwrap** the wrapped phase
  (ROMEO, PUROR, best-path) and **remove background fields** from air/bone/large
  vessels (PDF, SHARP, V-SHARP). We assume a clean local field map. Each of those
  stages is its own inverse problem.
- **Better inversion.** State-of-the-art dipole inversion is **MEDI**
  (Morphology-Enabled Dipole Inversion): an ℓ₁, edge-aware regularizer that ties
  the susceptibility gradient to the magnitude-image edges, plus **iLSQR** and
  total-variation variants. These couple voxels, so the FFT lives *inside* the
  iteration (the O(100)-FFT bottleneck the catalog highlights). Our Tikhonov
  gradient loop is the honest skeleton of that iteration, minus the spatial prior.
- **Deep-learning QSM.** **QSMnet / xQSM** replace the whole iterative solve with
  a single 3-D CNN forward pass (< 1 s), trained on COSMOS or MEDI labels. Same
  GPU, different math (a learned inverse instead of an explicit one).
- **Multiple orientations (COSMOS).** Scanning the head at several angles fills in
  the magic-angle cone from different directions and makes the inverse
  well-posed — the closest thing to a "gold-standard" χ, at the cost of extra
  scans.
- **Scale & precision.** Real volumes are 256³ at FP32, streamed with pinned
  memory, sometimes multi-GPU. Our 16×16×8 FP64 demo is instant and exact so the
  *mechanics* are legible; the physics and the k-space math are identical.

**Not for clinical use.** Synthetic phantom, simplified pipeline, educational
framing only (CLAUDE.md §8).

---

### Further reading (see README "Prior art")

MEDI toolbox (Cornell), ROMEO (phase unwrapping), QSMnet/xQSM (deep learning),
STISuite, and the QSM Reconstruction Challenge 2.0 for benchmark data and
reference reconstructions. Study these for the production approach; reimplement
didactically, do not copy wholesale (CLAUDE.md §2).
