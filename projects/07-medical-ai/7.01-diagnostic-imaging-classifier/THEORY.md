# THEORY — 7.1 Diagnostic Imaging Classifier

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._ This is a **reduced-scope teaching
> version** (CLAUDE.md §13): a convolutional-network **forward pass** with fixed,
> synthetic weights. §7 maps the honest gap to a real diagnostic system.

---

## 1. The science

Diagnostic imaging AI answers a deceptively simple question: *given a medical
image, is a pathology present?* A radiologist reading a chest X-ray or a CT lung
window is, in effect, computing a function from a grid of pixel intensities to a
label ("nodule", "no nodule", a disease grade, ...). Convolutional neural networks
(CNNs) learn to approximate that function from thousands of labeled examples
(MIMIC-CXR, CheXpert, LIDC-IDRI in the catalog).

Why *convolution*? Because the visual cue for a finding — the sharp edge of a
mass, the texture of ground-glass opacity, the round shape of a nodule — is a
**local pattern** that can appear **anywhere** in the image. A convolution slides
one small learned filter across the whole image, so the same pattern-detector is
applied at every location (**translation equivariance**). Stacking convolutions
builds up from edges → textures → parts → objects. The catalog's real backbones
(ResNet, EfficientNet, U-Net, ViT) are elaborations of this idea.

Our reduced-scope model keeps exactly the pieces that make a CNN a CNN and are
GPU-relevant: **convolution, a nonlinearity, pooling, and a linear classifier.**
We classify synthetic 16×16 patches into `normal` vs `lesion` (a bright central
blob standing in for a nodule). The "science" here is the *shape of the
computation*, not any clinical claim.

## 2. The math

**Inputs.** A batch of `n` grayscale images `x_i ∈ [0,1]^{H×W}` (here `H=W=16`),
and fixed weights: conv filters `w ∈ ℝ^{F×K×K}` with biases `b ∈ ℝ^F`
(`F=4`, `K=3`), and a dense layer `W_d ∈ ℝ^{C×M}`, `b_d ∈ ℝ^C` (`C=2` classes).

**Layer 1 — 2-D convolution (valid) + bias + ReLU.** For filter `f` and output
location `(oy, ox)` with `0 ≤ oy < H−K+1`, `0 ≤ ox < W−K+1`:

```
a_f(oy,ox) = ReLU( b_f + Σ_{ky=0}^{K-1} Σ_{kx=0}^{K-1} w_{f,ky,kx} · x(oy+ky, ox+kx) )
ReLU(t)    = max(0, t)
```

"Valid" padding means the `K×K` window stays fully inside the image, so the
feature map has size `(H−K+1)×(W−K+1) = 14×14`. This is `conv_pixel()` in
`reference_cpu.h`.

**Layer 2 — 2×2 max-pooling (stride 2).** For each filter, downsample by taking
the max over disjoint 2×2 blocks:

```
p_f(py,px) = max_{dy,dx ∈ {0,1}} a_f(2·py+dy, 2·px+dx)
```

giving `7×7` per filter. This is `pool_pixel()`. Flatten all filters' pooled maps
into one feature vector `feat ∈ ℝ^M`, `M = F·7·7 = 196`.

**Layer 3 — dense (fully connected) + softmax.** For class `c`:

```
z_c   = b_{d,c} + Σ_{j=0}^{M-1} W_{d,c,j} · feat_j          (dense_logit)
P(c)  = exp(z_c) / Σ_{c'} exp(z_{c'})                        (softmax)
pred  = argmax_c z_c
```

We report `P(lesion) = P(c=1)` and the `argmax` class. Softmax is computed in a
numerically stable way by subtracting `max(z_0, z_1)` before `exp` (`softmax_pos1`).

## 3. The algorithm

Straight-line forward pass, one image at a time on the CPU:

```
for each image i:
    for each filter f, out location (oy,ox):  conv_pixel  -> a_f(oy,ox)   # ReLU inside
    for each filter f, pooled  (py,px):        pool_pixel  -> feat[...]
    for each class c:                          dense_logit -> z_c
    pred[i] = argmax(z)
```

**Complexity.** The convolution dominates:

- Conv: `n · F · (H−K+1) · (W−K+1) · K²` multiply-adds
  = `4·4·14·14·9 ≈ 2.8·10⁴` MACs/image — trivially small here, but it scales as
  `O(n · F · H · W · K²)`, and for a real `512×512`, `F=64`, `K=3` layer that is
  `~4.8·10⁹` MACs per image.
- Pool: `O(n · F · (H/2) · (W/2))`, cheap.
- Dense: `O(n · C · M)`, cheap here.

**Data-access pattern.** Each conv output is a **gather**: it reads a small `K×K`
window of the input and a `K×K` slice of weights, and writes one scalar. Adjacent
outputs read overlapping windows — the reuse a real kernel exploits with tiling
(Exercise 2, and flagship `7.10`). Arithmetic intensity is low for `K=3` (few MACs
per byte loaded), which is why convolution is often memory-bandwidth-bound and why
`im2col`+GEMM / tensor cores (which reuse data through registers and shared memory)
win at scale.

**Parallelism.** Every conv output, every pooled value, and every logit is
**independent**. Work `= O(n·F·H·W·K²)`, but the *depth* (longest dependency
chain) is `O(K²)` for one dot product — i.e. essentially constant — so the problem
is embarrassingly parallel.

## 4. The GPU mapping

We use two kernels, mirroring the two arithmetic-heavy layers.

**Kernel 1 — `conv_pool_kernel` (one thread per pooled feature).** We launch
`n·M` threads (`M = F·POOL_H·POOL_W`). Thread `t` decodes a flat id into
`(image i, filter f, pooled py, pooled px)`, computes the four underlying
`conv_pixel` values for the 2×2 block, max-pools them, and writes **one** float.
Fusing conv+pool avoids materializing the full feature map in global memory.

**Kernel 2 — `dense_kernel` (one thread per (image, class)).** `n·C` threads;
thread `t → (i, c)` computes one `dense_logit` dot product.

**Thread → data map (kernel 1):**

```
t  = blockIdx.x * blockDim.x + threadIdx.x            # global thread id
px = t % POOL_W
py = (t / POOL_W) % POOL_H
f  = (t / (POOL_W*POOL_H)) % NUM_F
i  = t / FLAT                                         # FLAT = F*POOL_H*POOL_W
```

**Launch config.** Block = 256 threads (multiple of the 32-lane warp; 8 warps hide
latency; good occupancy on sm_75–sm_89). Grid = `ceil(total / 256)`; the ragged
last block is guarded by `if (t >= total) return;`.

**Memory hierarchy — and *why*:**

- **Constant memory** for all weights (`c_conv_w`, `c_conv_b`, `c_dense_w`,
  `c_dense_b`). Weights are small (< 1 KB here; 64 KB limit), read by every thread,
  and never written during a launch. Constant memory has a per-SM **broadcast
  cache**: when a warp reads the *same* address (e.g. the same filter tap), it is
  served in one transaction. Same idea as the query fingerprint in flagship `1.12`.
- **Global memory** for the image batch (input), the pooled features (scratch
  between kernels), and the logits (output). Reads of the input window are the
  bandwidth cost; a tiled version would stage windows into **shared memory**.
- **Registers** hold the running accumulator inside `conv_pixel`/`dense_logit`.
- **No atomics** — every thread owns a distinct output, so there are no races and
  the result is deterministic (see §5).

```
 grid of blocks (256 threads each) over n*FLAT pooled features
 ┌──────── block 0 ────────┐ ┌──────── block 1 ────────┐ ...
 │ t0  t1  t2 ...  t255    │ │ t256 ...          t511  │
 └───┬─────────────────────┘ └─────────────────────────┘
     │ decode t -> (i,f,py,px)
     ▼
   read 2x2 window of conv outputs, each = ReLU(bias + Σ w·x)   (w,b from __constant__)
     │ max-pool the four values
     ▼
   write ONE float  d_feat[i*FLAT + (f*POOL_H+py)*POOL_W + px]
 ────────────────────────────────────────────────────────────
   then dense_kernel: n*NUM_CLS threads, each a FLAT-length dot product -> one logit
```

**CUDA libraries (no black boxes).** Production convolution goes through **cuDNN**
(`cudnnConvolutionForward`), which internally picks an algorithm — often
**im2col + GEMM** (unfold each `K×K` window into a column, then one big
`cuBLAS` matrix multiply that keeps data in registers/shared memory and uses
**tensor cores** for FP16/BF16). We hand-write the naive gather instead so the math
is visible; Exercise 5 replaces the dense layer with a `cuBLAS Sgemm` to show the
GEMM view, and flagship `7.10` shows the shared-memory tiling that a fast conv uses.

## 5. Numerical considerations

- **Precision: FP32.** Neural-network inference is fine in single precision;
  real systems even drop to FP16/BF16 (mixed precision) or INT8 (TensorRT) for
  throughput, trading a little accuracy. We keep FP32 so the CPU reference and GPU
  match bit-for-bit.
- **Determinism.** No floating-point reduction is split across threads: each
  thread computes one output's dot product **sequentially, in the same order** as
  the CPU loop. There are no `atomicAdd` accumulations (which would reorder float
  sums and break reproducibility — see PATTERNS.md §3 and flagships `5.01`/`11.09`).
  So stdout is byte-identical every run, and Debug == Release.
- **ReLU / max ties.** `max(0,x)` and `max` of the 2×2 block use the same `>`
  comparison on both paths, so ties break identically.
- **Softmax stability.** Subtracting `max(z_0,z_1)` before `exp` prevents overflow
  for large logits; it does not change the ratio.

## 6. How we verify correctness

The GPU kernels and the CPU reference **call the same functions** — `conv_pixel`,
`pool_pixel`, `dense_logit`, `argmax2` — defined once in `reference_cpu.h` as
`__host__ __device__` inline functions (PATTERNS.md §2). The host compiler emits
them for `reference_cpu.cpp`; nvcc emits them for `kernels.cu`. Same source, same
operation order → **bit-identical FP32 results**.

Therefore the tolerance is **exactly `0`**: `main.cu` computes
`max |logit_cpu − logit_gpu|`, requires it to be `0`, and separately requires every
predicted class to match. On the committed sample this is `0.000e+00` (see the
`[verify]` stderr line). We *also* sanity-check the **science**: the two lesion
patches get `P(lesion) ≈ 1` and the two normal patches `P(lesion) < 0.5`, so batch
accuracy is 4/4 — the hand-designed Laplacian filter really does separate blobs
from flat tissue. Agreement between an obvious serial implementation and an
independent parallel one, *plus* recovering the planted labels, is convincing
evidence the kernels are right.

## 7. Where this sits in the real world

A production diagnostic classifier differs on every axis:

- **Training, not just inference.** Weights are *learned* by backpropagation (loss
  → gradients → optimizer), over augmented data (random affine/elastic transforms,
  the catalog's "AUC-optimised losses"). We ship fixed hand-designed weights.
- **Real backbones.** ResNet-50 / EfficientNet / 3D U-Net / ViT-B have tens of
  layers, residual connections, batch/layer norm, and attention — millions of
  parameters. We have one conv layer and one dense layer.
- **cuDNN + tensor cores + mixed precision.** Convolutions run through cuDNN with
  im2col+GEMM on tensor cores in FP16/BF16 (≈2× throughput). Deployment uses
  **TensorRT** INT8 for edge inference. Multi-GPU training synchronizes gradients
  with **NCCL all-reduce**; huge 3D volumes may need model parallelism.
- **Explainability & ensembling.** Grad-CAM highlights *why* a prediction was made;
  test-time augmentation (TTA) averages predictions over transformed inputs.
- **Real data & rigor.** DICOM I/O, intensity normalization, patient-level
  train/val/test splits, calibration, and regulatory validation — none of which a
  16×16 synthetic demo touches.

Study MONAI, TorchXRayVision, nnU-Net, and TotalSegmentator (README "Prior art")
to see the full pipeline. This project is the smallest honest version of the *inner
loop* that those systems accelerate.

---

## References

- **Krizhevsky, Sutskever, Hinton (2012)** — AlexNet; the conv→ReLU→pool→dense
  template this project distills.
- **He et al. (2015)** — ResNet; residual connections that make deep CNNs trainable.
- **Chetlur et al. (2014)** — *cuDNN: Efficient Primitives for Deep Learning*; how
  the convolution we hand-wrote is done fast (im2col + GEMM, algorithm selection).
- **MONAI** (https://github.com/Project-MONAI/MONAI) — medical-imaging framework;
  read its transforms and network zoo.
- **CheXpert** (Irvin et al., 2019) and **LIDC-IDRI** (Armato et al., 2011) — the
  real labeled datasets a trained version of this classifier would use.
- **NVIDIA CUDA C Programming Guide** — constant memory and the broadcast cache
  (§ on `__constant__`), which our weight storage relies on.
