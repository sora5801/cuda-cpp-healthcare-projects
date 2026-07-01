# THEORY — 4.4 Deep-Learning MRI/CT Reconstruction

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> **Reduced-scope teaching version.** This document explains both the small thing
> we *build* (an unrolled reconstruction with fixed operators) and the full thing
> it *models* (a trained E2E-VarNet). Where they differ, we say so.

---

## 1. The science

### Why MRI reconstruction is a computation at all

An MRI scanner does not photograph the body. It measures the **spatial Fourier
transform** of the tissue's magnetization — the *k-space*. Each readout samples
frequencies; the image is the inverse Fourier transform of a fully-sampled
k-space. So "reconstruction" literally means "inverse-transform the measurements".

The catch: filling k-space takes time, and time is the enemy (patient comfort,
motion, throughput, cost). **Accelerated MRI** deliberately **skips** k-space
lines. But an inverse transform of *incomplete* data is ill-posed: the missing
frequencies show up as **aliasing** (ghosts, wrap-around) and blur. You cannot
just invert; you must **reconstruct** — fill in the missing information using
prior knowledge of what anatomy looks like.

CT has the same shape of problem: low-dose or sparse-view acquisitions give noisy
or incomplete sinograms, and learned reconstruction cleans them up.

### From hand-crafted priors to learned priors

Classical compressed-sensing MRI solves

```
minimize  ||M·F·x − y||²  +  λ·R(x)
```

where `F` is the Fourier transform, `M` masks the sampled frequencies, `y` is the
measurement, and `R` is a hand-crafted regularizer (total variation, wavelet
sparsity). **Learned reconstruction** replaces `R` (or the whole solver's update)
with a **neural network trained on real anatomy**, so the prior is *data-driven*
instead of guessed. E2E-VarNet does this by **unrolling** the iterative solver
into network layers.

---

## 2. The math

### The forward model

Let `x ∈ ℝ^{ny×nx}` be the image and `F` the 2-D DFT. The scanner measures

```
y = M ⊙ (F x)          (⊙ = elementwise; M ∈ {0,1} is the sampling mask)
```

Unsampled frequencies are simply absent (we store them as 0).

### The variational objective

We seek the image that both fits the measurement and looks like plausible anatomy:

```
x* = argmin_x  ½ ||M ⊙ (F x) − y||²  +  λ R(x)
```

A **proximal-gradient** step splits this into two moves per iteration:

- a **regularizer / denoiser** move (the prox of `λR`): `x ← prox_{λR}(x)`, and
- a **data-consistency** move (the gradient of the data term): pull `Fx` back
  toward `y` on the sampled frequencies.

### The two operators we actually implement

**Denoiser move (image domain).** We use a damped step toward a smoothed image:

```
x_new(p) = x(p) + λ ( D(x)(p) − x(p) )
```

`D` is a fixed normalized 3×3 Gaussian (the 1-2-1 outer product / 16), applied
with clamp (replicate-edge) boundaries. `D` is a genuine low-pass denoiser: it
suppresses the high-frequency aliasing that under-sampling injects. In a trained
network `D` is a CNN and `λ` is learned per stage; here both are fixed constants
(`recon_core.h`).

**Data-consistency move (frequency domain).** For a *hard* data-consistency
projection (the simplest and most stable form),

```
K = F x                          (forward transform the current estimate)
K(k) = y(k)   where M(k) = 1     (overwrite sampled bins with the measurement)
x = F⁻¹ K                         (inverse transform)
```

This guarantees the estimate always reproduces the data we actually measured; the
denoiser is only allowed to *invent* the **unsampled** frequencies.

### The DFT pair we hand-roll

With an **un-shifted** DFT (DC at index 0, low frequencies wrapping to the array
ends — which is why our mask keeps a band at *both* ends of each axis):

```
Forward   F[v,u] = Σ_{y,x} x[y,x] · exp(−2πi (v y/ny + u x/nx))
Inverse   x[y,x] = (1/N) Σ_{v,u} F[v,u] · exp(+2πi (v y/ny + u x/nx))      N = ny·nx
```

`F⁻¹(F x) = x` to floating-point precision — the identity the data-consistency
step relies on.

---

## 3. The algorithm (and complexity)

```
recon(y, M, stages T, λ):
    x ← F⁻¹(y)                         # zero-filled init (aliased start image)
    repeat T times:
        d ← denoise_step(x, λ)         # image-domain 3x3 stencil over all pixels
        K ← F(d)                       # forward DFT
        K ← data_consistency(K, y, M)  # overwrite sampled bins with measurement
        x ← F⁻¹(K)                     # inverse DFT
    return x
```

**Complexity (this teaching build, direct DFT).**

| Step | Serial cost | Notes |
|------|-------------|-------|
| denoiser stencil | `O(N)` (9 taps/pixel) | `N = ny·nx` |
| forward / inverse DFT | `O(N²)` each | direct transform |
| per stage | `O(N²)` | DFT dominates |
| whole recon | `O(T·N²)` | |

**Complexity (production).** With `cuFFT` the transforms are `O(N log N)`, so a
stage is `O(N log N)` plus the CNN's `O(N·k²·C²)` convolution cost. This is why
real pipelines are fast and ours is only tractable on a tiny image.

The **parallel depth** is what matters for the GPU: each stage's kernels are
`O(1)` depth (all outputs independent), so the wall-clock is (stages × a few
launches), not (stages × N).

---

## 4. The GPU mapping

Two independent, embarrassingly-parallel patterns, one per operator.

### Pattern A — the denoiser: a per-pixel **stencil**

```
regularize_kernel:  one thread per pixel (x,y)
  block = 16 x 16 threads   (256 = warp multiple, good occupancy on sm_75..89)
  grid  = ceil(nx/16) x ceil(ny/16) blocks
  thread (x,y) = (blockIdx.x*16+threadIdx.x, blockIdx.y*16+threadIdx.y)
  reads its 3x3 neighbourhood of `in` (global), writes one pixel of `out`
```

Neighbouring threads re-read overlapping pixels. At this size the L1/L2 cache
absorbs the reuse, so we keep the code simple; the classic optimization (stage a
tile + halo in **shared memory**) is spelled out in flagship 7.10 and left as an
exercise. No atomics: outputs are independent.

### Pattern B — the transforms: a per-output **gather + reduce**

```
dft_forward_kernel:  one thread per OUTPUT frequency (v,u)
dc_idft_kernel:      one thread per OUTPUT pixel (y,x)
  each thread reduces over ALL input elements -> one output value
```

Each output is a full independent reduction — a clean map of "one thread per
output". Complex numbers are stored as **two float arrays** (`re[]`, `im[]`,
structure-of-arrays) so consecutive threads read consecutive memory (coalesced).

### The unroll loop and **ping-pong** buffers

The host drives the stages (like flagship 6.04). Kernels A and C each read one
image buffer and write another, so we alternate two device buffers (`cur`/`nxt`)
— never writing a buffer we are also reading. The measured k-space and mask are
uploaded **once** (constant across the recon). Data flow per stage:

```
cur --regularize--> nxt --forwardDFT--> (kre,kim) --dc_apply--> (kre,kim) --iDFT--> cur
```

### Memory hierarchy used

| Data | Space | Why |
|------|-------|-----|
| image buffers, k-space | **global** | large, read/written across the grid |
| measured k-space, mask | **global** (uploaded once) | constant during the recon |
| per-thread accumulators (`re`,`im`,`acc`) | **registers** | fast, private |
| (exercise) 3×3 tile | **shared** | reuse across a block's threads |

A trained pipeline additionally leans on **tensor cores** (FP16/BF16 GEMMs inside
the CNN) and `cuDNN` for the convolutions — the same stencil shape, hardware-tuned.

---

## 5. Numerical considerations

- **Precision.** We compute in **FP32**, matching real-time reconstruction (which
  uses FP16/FP32 mixed precision on tensor cores). Angles use `cosf/sinf`.
- **CPU vs GPU agreement.** Both paths call the **same** `__host__ __device__`
  functions (`recon_core.h`, `dft_core.h`), accumulating in the **same order**
  (y-outer, x-inner). So they would be bit-identical *except* for **fused
  multiply-add (FMA) reassociation**: the device (and the host compiler at `-O2`)
  fuse `a*b+c` differently than a strict left-to-right evaluation. Over `T` stages
  × two DFTs, this drifts by `~1e-5` on pixel values in `[0,1]`. We therefore
  verify to a **physically-negligible `1e-3`** and *say so* rather than pretending
  bit-exactness (PATTERNS.md §4). The observed `max_abs_err ≈ 8e-6` is far inside
  it.
- **Determinism of stdout.** No atomics anywhere (every output element is
  independent), so there is **no order-dependent floating-point sum** — the result
  is reproducible run-to-run. We print the pixel fingerprint at **4 decimals**
  because the 5th–6th digits wobble between Debug (`-G`) and Release (`-lineinfo
  -O2`) builds due to the FMA difference above; 4 decimals is stable across both.
  Timings and the exact error go to **stderr** (shown, not diffed).
- **Division guards.** The denoiser normalizer (16) and the iDFT normalizer (`N`)
  are compile-time positive; no divide-by-zero path exists.

---

## 6. How we verify correctness

Two independent checks:

1. **GPU == CPU (implementation correctness).** `main.cu` runs `recon_cpu` and
   `recon_gpu` on the identical `Acquisition` and asserts
   `max_abs_err(GPU, CPU) ≤ 1e-3`. Because both share the per-element cores, this
   isolates *executor* bugs (indexing, launch config, buffer swaps) from *math*
   bugs.
2. **Science score (does it actually reconstruct?).** We compare the RMS error to
   the ground-truth phantom for the **zero-filled** image vs the **reconstructed**
   image. The unroll must *lower* it (here ~11%). This is the analytic-style,
   "validate the science not just the agreement" check PATTERNS.md §4 recommends:
   a recon that matched the CPU but did not beat zero-filling would be a correct
   implementation of a useless method — this catches that.

Edge cases exercised by the synthetic sample: DC/low-frequency band always kept
(so the recon is well-posed), a mix of sampled/skipped high frequencies (so the
denoiser has something to fill), and a phantom with both a curved edge (disk) and
straight edges (square) so smoothing artifacts are visible.

---

## 7. Where this sits in the real world

| Aspect | This teaching demo | Production (E2E-VarNet / fastMRI) |
|--------|--------------------|-----------------------------------|
| Denoiser `D` | fixed 3×3 Gaussian | **trained CNN** (U-Net refinement per stage) |
| `λ`, #stages | fixed constants | **learned** end-to-end |
| Coils | single, real-valued | **multi-coil complex**, with learned sensitivity maps |
| Transform | direct `O(N²)` DFT | **`cuFFT`** `O(N log N)` |
| Data consistency | hard mask-replace | soft gradient step (learned weight) |
| Precision / HW | FP32 | FP16/BF16 mixed on **tensor cores**, `cuDNN` |
| Training | none | batch SGD on TB of raw k-space, **multi-GPU DDP / NCCL** |
| Runtime | ms on a toy image | **sub-100 ms** per 256²×32-coil volume on an A100 |

**How to grow this into the real thing (the seams are deliberate):**

- Replace `denoise_pixel` in `recon_core.h` with a learned CNN — the stencil is
  already the CNN's access pattern; `cuDNN` slots in here.
- Replace the DFT kernels with `cuFFT` R2C/C2R (see flagship 8.03 for the `cuFFT`
  idiom) — the single biggest speedup.
- Add coils + sensitivity maps and a soft (weighted) data-consistency step to
  match E2E-VarNet exactly.
- Add a training loop (autograd) in PyTorch and learn `λ`, the stage count, and
  the CNN weights on fastMRI.

Everything above is why this is filed under **Active R&D**: the *structure* is
settled and shown here; the *learning* is where the field is still moving
(diffusion-model priors, plug-and-play denoisers, recurrent unrolls).

---

### Further reading

fastMRI baseline (E2E-VarNet, U-Net) · DIRECT (many unrolled architectures) · BART
(classical + deep, done rigorously). Links in [README.md](README.md#prior-art--further-reading).
