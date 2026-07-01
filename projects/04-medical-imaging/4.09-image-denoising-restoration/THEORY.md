# THEORY — 4.9 Image Denoising & Restoration (Non-Local Means)

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

Every medical image is a measurement, and every measurement carries **noise**.
Where the noise comes from depends on the modality:

- **CT / X-ray / PET — quantum (Poisson) noise.** The image is built from a finite
  number of detected photons. Fewer photons → grainier image. *Low-dose* CT
  deliberately uses fewer photons to reduce the patient's radiation exposure, and
  pays for it with noise. Denoising is what makes low-dose CT diagnostically usable
  — it is one of the most active areas in medical imaging.
- **MRI — thermal noise.** Random thermal motion in the body and the receive coils
  adds (approximately) Gaussian noise to the complex k-space signal, which becomes
  **Rician** noise in the reconstructed magnitude image.
- **Ultrasound — speckle.** Coherent interference of the ultrasound wavefront gives
  a multiplicative, texture-like noise.

The clinical tension is always the same: **remove the noise without removing the
signal.** A naive blur (Gaussian filter) lowers noise but also smears the edges of
a tumour, a vessel wall, or a fracture line — exactly the features a radiologist
needs. A good denoiser is *edge-preserving*: it smooths flat tissue while keeping
boundaries sharp.

**Non-Local Means (NLM)** is the classic edge-preserving denoiser, and the one we
build here. Its insight is beautifully simple: to estimate a pixel's true value,
average it with *other pixels that sit in a similar local context* — not just its
spatial neighbours. Two points on the same tissue boundary "look alike" in their
surrounding patch even if they are far apart in the image, so they are good
evidence for each other. Averaging over such patch-similar pixels cancels the
(zero-mean) noise while leaving the underlying structure intact.

## 2. The math

Let the noisy image be `v(p)` for pixel position `p = (row, col)`, with intensities
normalized to `[0, 1]`. NLM estimates the denoised image `u(p)` as a weighted
average over a **search window** `N(p)` of candidate pixels `q` around `p`:

```
              Σ_{q ∈ N(p)}  w(p, q) · v(q)
   u(p)  =   ───────────────────────────────
              Σ_{q ∈ N(p)}  w(p, q)
```

The weight `w(p, q)` measures how similar the **patches** around `p` and `q` are.
A patch `P(p)` is the small square of `(2R+1)²` pixels centred at `p` (`R` = patch
radius). The squared patch distance is the mean squared per-pixel difference:

```
   d²(p, q) = (1 / |P|) · Σ_{k ∈ patch offsets}  ( v(p+k) − v(q+k) )²
```

`d²` has units of intensity² and is `≥ 0`; identical patches give `0`. The weight
turns distance into similarity with a Gaussian falloff:

```
   w(p, q) = exp( − max( d²(p, q) − 2σ², 0 ) / h² )
```

| Symbol | Meaning | Units / range |
|---|---|---|
| `v, u` | noisy input, denoised output | intensity, `[0,1]` |
| `R` | patch radius; patch is `(2R+1)²` px | pixels (e.g. 2 → 5×5) |
| `S` | search radius; window is `(2S+1)²` px | pixels (e.g. 5 → 11×11) |
| `σ` | noise standard deviation | intensity |
| `h` | filter strength (decay scale) | intensity; often `h ≈ 1.2σ` |
| `d²(p,q)` | mean squared patch difference | intensity² |
| `w(p,q)` | similarity weight | `(0, 1]` |

Two subtleties that matter:

- **The `2σ²` noise-bias correction.** Even two patches of the *same* underlying
  signal differ by an *expected* `2σ²` in mean-squared terms purely because each is
  independently corrupted by variance-`σ²` noise. Subtracting `2σ²` (clamped at 0)
  removes that expected floor so genuinely-identical structure gets weight ≈ 1.
- **`h` sets how fast similarity decays.** Small `h` → only near-identical patches
  contribute → less smoothing, more residual noise. Large `h` → flatter weights →
  more smoothing, risk of blurring. This is the one knob a user tunes.

The **quality metric** is PSNR (peak signal-to-noise ratio) against the clean image:
`MSE = mean((u − clean)²)`, `PSNR = 10·log₁₀(1² / MSE)` dB (peak = 1 for `[0,1]`
images). A working denoiser raises PSNR relative to the noisy input.

## 3. The algorithm

```
for each output pixel p = (row, col):        # P pixels
    wsum = 0 ; vsum = 0
    for each candidate q in the (2S+1)² search window around p:
        d2 = 0
        for each of the (2R+1)² patch offsets k:      # patch compare
            d2 += (v(p+k) − v(q+k))²
        d2 /= (2R+1)²
        w   = exp(−max(d2 − 2σ², 0) / h²)
        wsum += w
        vsum += w · v(q)
    u(p) = vsum / wsum
```

**Complexity.** Serial cost is `O( P · (2S+1)² · (2R+1)² )`. For a 512×512 slice
with `S=10, R=3`, that is `262144 · 441 · 49 ≈ 5.7×10⁹` patch-pixel operations —
seconds on a CPU, and it scales *quadratically* in both window and patch size.

**Parallelism.** The key structural fact: **the outer loop over output pixels is
fully independent.** `u(p)` reads only the (read-only) noisy input and writes only
its own output. No pixel depends on another's result. So the algorithm has:
- **work** = `O(P·S²·R²)` (same as serial — we do not reduce total work), and
- **depth** = `O(S²·R²)` (one pixel's inner loops), independent of `P`.

That is the definition of embarrassingly parallel: give each of the `P` output
pixels its own thread and they all run at once. **Arithmetic intensity** is high
(each loaded input pixel participates in many patch comparisons across overlapping
patches and overlapping search windows), and neighbouring output pixels read
heavily overlapping regions — which is why the naive kernel is already cache-friendly.

## 4. The GPU mapping

**Thread-to-data mapping.** One thread owns one output pixel. We tile the 2-D image
with a 2-D grid of 2-D blocks:

```
   block = 16 × 16 threads (= 256)
   grid  = ceil(width/16) × ceil(height/16) blocks

   thread (blockIdx, threadIdx) → pixel:
       col = blockIdx.x·blockDim.x + threadIdx.x
       row = blockIdx.y·blockDim.y + threadIdx.y
   (threads with row/col outside the image return immediately — the ragged-edge guard)
```

```
        image (width × height)                  one block's threads
   ┌───────────────────────────────┐            ┌───────────────┐
   │ ┌────┐┌────┐┌────┐            │            │ t t t t ...   │  16 threads wide
   │ │blk ││blk ││blk │  ...        │            │ t t t t       │  16 threads tall
   │ └────┘└────┘└────┘            │            │ t t t t       │  = 256 threads
   │ ┌────┐┌────┐                  │            │ ...           │  each = 1 output pixel
   │ │blk ││blk │   16×16 tiles     │            └───────────────┘
   │ └────┘└────┘                  │
   └───────────────────────────────┘
```

**Why 16×16 = 256 threads/block.** 256 is a multiple of the 32-lane warp (no
wasted lanes) and gives the scheduler 8 warps per block to hide the latency of the
global-memory patch reads. A *square* tile matches the 2-D image so threads in a
block touch neighbouring pixels — their patch and search reads overlap heavily, so
the L1/L2 caches serve most of them.

**Memory hierarchy.**
- **Global memory:** the noisy image, uploaded once, read by every thread. This is
  the only large buffer. Because access is spatially local and overlapping, the
  hardware caches carry the reuse for us in this teaching version.
- **Registers:** the two running accumulators (`Σw`, `Σw·v`) and the loop indices
  live entirely in registers inside `nlm_pixel()`. Nothing is shared *between*
  threads, so there is **no `__shared__` memory and no atomics** — output pixels are
  independent, so there is nothing to synchronise or reduce across threads.
- **Constant memory:** not needed here; the 6 scalar parameters are passed by value
  in the `NlmParams` struct, so each thread has its own register copy.

**The obvious next optimisation (left as an exercise).** The naive kernel re-reads
overlapping input from global memory across neighbouring threads. A block could
cooperatively stage its input **tile plus a halo** of width `S+R` into `__shared__`
memory once, then every thread reads patches from fast shared memory. This is the
same shared-memory-tiling idiom as the 1-D convolution flagship (`7.10`), extended
to 2-D. Two further speed-ups: (1) an *early abort* that stops accumulating a patch
distance once it exceeds a threshold (the weight will be ≈ 0 anyway); (2) an
**integral image** so each patch distance is `O(1)` instead of `O(R²)`.

**No CUDA library is used.** NLM block matching is a hand-written kernel — that is
the whole point of the "custom CUDA" pattern in the catalog. (The *learned*
denoisers in the same catalog entry — DnCNN, RED-CNN — would use **cuDNN** for their
convolutions and **cuBLAS** for dense layers, and **TensorRT** for FP16 deployment;
see §7. Writing a cuDNN convolution by hand means the tiled-GEMM/implicit-GEMM
kernels those libraries hide — a project in its own right.)

## 5. Numerical considerations

- **Precision: FP32.** Images are `[0,1]` and 8–16 bit in origin, so single
  precision is far more than enough; FP32 also doubles throughput and halves memory
  vs FP64. All accumulation happens in `float`. The PSNR metric itself accumulates
  MSE in `double` so the *reported quality number* does not lose accuracy on large
  images — but that is a host-side metric, not the kernel.
- **No race conditions, no atomics.** Each thread writes exactly one distinct output
  pixel and reads only the immutable input. There is nothing to race on, so no locks
  or atomics appear anywhere.
- **Determinism.** Because there is no cross-thread reduction, the per-pixel sum is
  computed in a *fixed* order (the same nested loop on CPU and GPU), so there is no
  atomic-reordering nondeterminism (contrast the Monte-Carlo `5.01` and k-means
  `11.09` flagships, which must accumulate in integers to stay deterministic). The
  only CPU↔GPU difference is **FMA contraction**: the GPU may fuse `a*b + c` into one
  rounding step where the host emits two. Over the thousands of multiply-adds per
  pixel this reaches `~2×10⁻⁷` in float — real, small, and honestly reported.
- **Borders.** Patches near the image edge reach out of bounds. We **mirror**
  (reflect) coordinates back inside — smooth (no hard seam) and, crucially, written
  once in the shared core so CPU and GPU border-handle identically.
- **Divide-by-zero.** The centre candidate `q = p` always has `d² = 0 → w = 1`, so
  `Σw > 0` and the final division is always safe; we still guard it defensively.

## 6. How we verify correctness

Two independent checks:

1. **GPU vs CPU agreement (the correctness gate).** `src/reference_cpu.cpp` computes
   the whole image with a plain double loop; `src/kernels.cu` computes it on the GPU.
   Both call the **same** `nlm_pixel()` from `src/nlm_core.h` (compiled once for the
   host, once for the device via the `__host__ __device__` idiom), so the arithmetic
   is identical up to FMA contraction. `main.cu` computes `max_abs_err` over all
   pixels and requires it `≤ 1.0e-4`. Observed on an RTX 2080 (sm_75): `2.4e-7` —
   three orders of magnitude inside tolerance. The tolerance is set to a small
   *absolute* value (not exact 0) precisely because of FMA (PATTERNS.md §4); `1e-4`
   intensity is ~0.03 of one 8-bit grey level — invisible. An independent serial
   implementation agreeing with the parallel one to machine-ε is strong evidence
   both are correct: a bug would have to be identical in two very differently-shaped
   pieces of code.
2. **The science actually works (a stronger check).** We measure PSNR of the noisy
   input and of the denoised output against the synthetic clean image. On the sample
   the denoiser lifts PSNR from **22.09 dB → 29.99 dB (+7.90 dB)**, and the recovered
   central-row profile matches the phantom's dark field, bright disk, and darker
   square. That validates the *algorithm*, not just CPU==GPU agreement.

**Edge cases exercised:** border pixels (mirrored patches), the flat background
(should smooth strongly), and the two edge types (disk arc + square) where over-
smoothing would be visible in the row profile.

## 7. Where this sits in the real world

This is a **reduced-scope teaching version** (CLAUDE.md §13): classic NLM, the
method that most cleanly demonstrates the catalog's "custom CUDA block matching"
pattern with no trained weights and no external library. The broader field:

- **BM3D / BM4D** — the strongest *classical* denoisers. They extend NLM by *block
  matching* similar patches into a 3-D stack, then collaboratively filtering the
  stack in a transform domain (DCT/wavelet) with shrinkage, and aggregating back.
  Same block-matching core we build here, plus a transform and a Wiener stage.
- **DnCNN** — a CNN that learns to *predict the noise residual* `v − u`; subtract it
  to denoise. On the GPU its convolutions run through **cuDNN**.
- **RED-CNN** — an encoder–decoder CNN tuned specifically for **low-dose CT**, in the
  MONAI model zoo; the clinically-framed successor to this project.
- **Noise2Void / Noise2Self** — *self-supervised*: they train with no clean targets
  by masking pixels and predicting them from context, which is essential in
  microscopy and MRI where clean ground truth does not exist.
- **Score-based diffusion (DDPM/DDIM, DiffusionMBIR)** — current state-of-the-art for
  *perceptual* quality. They learn a denoiser at many noise levels and run 50–1000
  reverse-diffusion steps, each a full UNet forward pass — GPU-mandatory, and often
  used as a *prior* inside iterative CT reconstruction rather than as a standalone
  filter.

What the learned methods add over NLM: they capture *learned* image statistics
(anatomy priors) rather than only self-similarity, so they denoise harder cases
better — at the cost of training data, trained weights, and cuDNN/TensorRT infra.
What NLM keeps: it is *training-free*, interpretable, and a perfect first CUDA
denoising kernel.

---

## References

- **A. Buades, B. Coll, J.-M. Morel (2005), "A non-local algorithm for image denoising", CVPR.** The paper this kernel implements; read it for the weight derivation and the `2σ²` correction.
- **K. Dabov et al. (2007), "Image denoising by sparse 3-D transform-domain collaborative filtering" (BM3D).** How block matching plus transform-domain shrinkage beats plain NLM.
- **K. Zhang et al. (2017), "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN (DnCNN)"** — [cszn/DnCNN](https://github.com/cszn/DnCNN). The learned residual-denoising successor.
- **H. Chen et al. (2017), "Low-Dose CT with a Residual Encoder-Decoder CNN (RED-CNN)"** — in the MONAI model zoo. The low-dose-CT clinical framing.
- **A. Krull et al. (2019), "Noise2Void"** — [juglab/n2v](https://github.com/juglab/n2v). Self-supervised denoising without clean targets.
- **DiffusionMBIR** — [HJ-harry/DiffusionMBIR](https://github.com/HJ-harry/DiffusionMBIR). Score-based diffusion as a reconstruction/denoising prior for CT.
- **2016 AAPM Low-Dose CT Grand Challenge** — https://www.aapm.org/grandchallenge/lowdosect/ — the quarter-dose/full-dose paired CT data these methods are benchmarked on.
