# THEORY — 7.18 Retinal Fundus AI Screening

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

---

## 1. The science

The **fundus** is the interior back surface of the eye — the retina, optic disc,
macula, and blood vessels — visible through the pupil with a fundus camera. It is
the one place in the body where you can directly image blood vessels and nerve
tissue non-invasively, which makes it a rich diagnostic window.

**Diabetic retinopathy (DR)** is progressive damage to the retinal
microvasculature caused by chronic high blood sugar. Clinicians grade it on a
5-point scale:

| Grade | Name | Hallmark findings |
|------:|------|-------------------|
| 0 | No DR | normal fundus |
| 1 | Mild | a few *microaneurysms* (tiny red dots) |
| 2 | Moderate | more microaneurysms, dot/blot *haemorrhages*, hard *exudates* (bright spots) |
| 3 | Severe | extensive haemorrhages, venous beading |
| 4 | Proliferative | neovascularisation (new fragile vessels), risk of blindness |

DR is a leading cause of preventable blindness, and screening is high-volume: a
clinician (or an AI) must scan **millions of images per year**. That volume, plus
the fact that early DR is subtle, is exactly why automated CNN screening —
validated systems like IDx-DR and the Google/DeepDR work — became one of the
first FDA-cleared medical-AI applications. The same pipeline generalises to
glaucoma (optic-disc cupping) and age-related macular degeneration (macular
drusen).

**What the model keys on** is local visual structure: small red blobs
(microaneurysms), bright spots (exudates), vessel morphology, and colour
contrast. Convolutional filters are the natural tool because these are
*translation-invariant local patterns* — a microaneurysm looks the same wherever
it appears.

---

## 2. The math

### 2.1 The image

A colour fundus image is a tensor `x ∈ [0,1]^{C×H×W}` with `C=3` (R,G,B). We
store it **channel-major** (planar): `x[c,y,x] = data[c·H·W + y·W + x]`.

### 2.2 Convolution (the core operation)

A convolution layer with `C_out` filters maps an input stack of `C_in` channels
to an output stack of `C_out` feature maps. For output channel `oc` at pixel
`(y,x)`:

```
out(oc, y, x) = b[oc] + Σ_{ic=0}^{C_in-1} Σ_{ky=0}^{K-1} Σ_{kx=0}^{K-1}
                        W[oc, ic, ky, kx] · in(ic, y+ky-h, x+kx-h)
```

where `K` is the kernel size (here 3), `h = (K-1)/2 = 1` is the halo, and inputs
outside `[0,H)×[0,W)` are treated as zero ("same" padding). `W` has shape
`C_out×C_in×K×K`; `b` has shape `C_out`. This is exactly `conv_at()` in
[`src/cnn_core.h`](src/cnn_core.h).

### 2.3 The nonlinearity, pooling, and head

- **ReLU:** `relu(x) = max(0, x)`. Without a nonlinearity, stacked convolutions
  would collapse into a single linear map.
- **2×2 max-pool (stride 2):** `pool(c,oy,ox) = max` over the `2×2` window at
  `(2·oy, 2·ox)`. Halves `H` and `W`; adds a little translation tolerance.
- **Global average pool (GAP):** `gap[c] = mean_{y,x} f(c,y,x)` — collapses each
  final feature map to one number, giving a `C2`-dimensional embedding.
- **Fully-connected classifier:** `logit[k] = b_fc[k] + Σ_c W_fc[k,c]·gap[c]`.
- **Softmax:** `p[k] = exp(logit[k]) / Σ_j exp(logit[j])`, computed stably by
  subtracting `max_j logit[j]` first. `argmax_k p[k]` is the predicted grade.

### 2.4 The whole network

```
x (3×H×W)
  → Conv3×3(3→6)  → ReLU → MaxPool2×2   →  f1 (6 × H/2 × W/2)
  → Conv3×3(6→12) → ReLU → MaxPool2×2   →  f2 (12 × H/4 × W/4)
  → GAP                                  →  g  (12)
  → FC(12→5) → Softmax                   →  p  (5) → argmax = DR grade
```

---

## 3. The algorithm & complexity

The forward pass is a straight-line sequence of layers (no iteration to
converge). The dominant cost is the convolutions.

**Per conv layer:** `C_out · H · W · C_in · K²` multiply-accumulates. For layer 1
on a 32×32 image: `6·32·32·3·9 ≈ 166k` MACs; layer 2 (on 16×16): `12·16·16·6·9 ≈
166k`. On a real 2048×2048 image with a deep backbone this reaches **billions** of
MACs per image — hence the GPU.

- **Serial (CPU):** `O(C_out·H·W·C_in·K²)` time per layer, done with nested loops
  (`conv_relu_pool_cpu` in `reference_cpu.cpp`). Total across the net is the sum
  over layers.
- **Parallel (GPU):** the *same* total work, but the `C_out·H·W` output pixels are
  computed concurrently. Ideal span is `O(C_in·K²)` (one thread's inner sum) plus
  the pooling/GAP reductions. Real speed is bounded by memory bandwidth, which is
  why the tiling in §4 matters.

Pooling and CAM are `O(C·H·W)`; GAP is a reduction of `O(C·H·W)`; the FC head is
trivially `O(NUM_CLASSES·C2)`.

---

## 4. The GPU mapping

### 4.1 Threads, blocks, grid

For the conv and pool kernels the launch is 3-D:

```
block  = (TILE, TILE)           // 16×16 = 256 threads (good occupancy, sm_75..89)
grid   = (⌈W/TILE⌉, ⌈H/TILE⌉, C_out)
```

- `blockIdx.z` selects the **output channel** `oc`.
- Thread `(threadIdx.x, threadIdx.y)` in block `(blockIdx.x, blockIdx.y)` owns
  output pixel `(x = blockIdx.x·TILE + threadIdx.x, y = blockIdx.y·TILE + threadIdx.y)`.

### 4.2 Why shared memory (the headline lesson)

Each output pixel reads a `3×3` window of *every* input channel. Adjacent output
pixels overlap by two of three rows/columns, so a naive "read from global memory"
kernel re-reads each input pixel up to `K² = 9` times. Global-memory bandwidth is
the scarce resource on a GPU, so that redundancy caps performance.

The fix is **tiling**: each block cooperatively loads its
`(TILE+2h)×(TILE+2h)` haloed patch of one input channel into **shared memory**
(fast on-chip, ~100× lower latency than global) *once*, `__syncthreads()`, then
every thread reads its `3×3` window from the tile. We loop this over input
channels, accumulating into a per-thread register. This is the 2-D generalisation
of the 1-D tiling in **flagship 7.10**; see [`src/kernels.cu`](src/kernels.cu),
`conv_relu_tiled`.

```
   global input channel                shared tile (TILE+2h)²         thread's 3×3 window
   ┌───────────────┐   cooperative     ┌───────────────┐   read       ┌─────┐
   │ . . . . . . . │   strided load    │ h h h h h h h │  from SMEM →  │ w w w│
   │ . ┌───────┐ . │  ───────────────▶ │ h ┌───────┐ h │               │ w w w│
   │ . │ tile  │ . │                   │ h │ tile  │ h │               │ w w w│
   │ . └───────┘ . │                   │ h └───────┘ h │               └─────┘
   └───────────────┘                   └───────────────┘
```

### 4.3 The memory hierarchy used

- **Global memory:** the image, weights, and every intermediate feature stack.
- **Shared memory:** the per-channel input tile in the conv kernel; the partial
  sums in the GAP block reduction.
- **Registers:** each thread's convolution accumulator `acc`.
- **Constant memory:** *not* used here (the weights are large and vary per output
  channel, unlike 7.10's tiny shared FIR); a natural exercise is to cache small
  weight blocks in shared memory too.

### 4.4 The other kernels

- `maxpool2x2` — one thread per output pixel; embarrassingly parallel.
- `global_avg_pool` — **one block per feature channel**; the block's threads
  stride over the `H·W` plane summing into shared memory, then a classic
  tree reduction (`for stride = blockDim/2; stride>0; stride>>=1`) folds it to a
  single sum. Thread 0 divides by `H·W`.
- `fc_logits` — `NUM_CLASSES` threads, one logit each.
- `cam_kernel` — one thread per CAM pixel.

Softmax + argmax (5 numbers) are done on the host — not worth a launch.

---

## 5. Numerical considerations

- **Precision:** FP32 throughout, matching real inference (GPUs run FP32/FP16 for
  CNNs; FP64 is pointless here). The CPU reference also uses FP32 for the conv MAC
  via the shared `conv_at()` — so those match to the last bit given the same
  summation order.
- **The one real difference — reduction order.** The CPU's global-average-pool
  sums the `H·W` plane serially in `double`; the GPU sums it with a `float` tree
  reduction. Floating-point addition is **not associative**, so the two totals
  differ by a few ULPs. That propagates through the FC layer into the logits,
  giving a `max_abs_err ≈ 1e-8` (observed) — far below any decision boundary.
- **Determinism.** Every kernel here writes each output exactly once (no
  `atomicAdd` into shared accumulators), and the tree reduction has a fixed order,
  so the GPU result is **bit-identical run to run** (PATTERNS.md §3). That is why
  `stdout` can be diffed against `expected_output.txt`; timings go to `stderr`.
- **No fast-math.** `--use_fast_math` is off (see the `.vcxproj`) so `exp()` in
  softmax and the MACs are IEEE-accurate.

---

## 6. How we verify correctness

Two independent checks (PATTERNS.md §4):

1. **CPU == GPU.** `main.cu` runs `forward_cpu` and `forward_gpu` — which call the
   *same* `cnn_core.h` math — and compares `logits`, `probs`, and the Grad-CAM
   `cam` with `util::max_abs_err`. Tolerance is **`1e-3`**, a small physical
   epsilon chosen to comfortably cover the reduction-order divergence of §5 (the
   actual error is ~`1e-8`). We also assert the **argmax grade matches** — the
   thing a screening system actually outputs.
2. **Softmax sanity.** The probabilities are a valid distribution (sum to 1,
   non-negative). Because the softmax is shared, equal logits give equal probs.

**Grad-CAM here** is the *class-activation-mapping* simplification: for the
winning class `best`, `CAM(y,x) = ReLU(Σ_c W_fc[best,c] · f2(c,y,x))`. It
highlights the regions of the final feature grid that pushed the winning class
up — a coarse "where did the model look?" map, our stand-in for lesion
localisation. True Grad-CAM weights each feature map by the *gradient* of the
class score w.r.t. that map; with GAP + a linear head the two coincide up to a
constant, which is why CAM is a faithful teaching version (see the Grad-CAM
paper, Selvaraju et al. 2017). Implementing the gradient version is Exercise 4.

---

## 7. Where this sits in the real world

This is a deliberately **reduced-scope teaching version** (CLAUDE.md §13). What a
production DR-screening model changes — none of it changes the *arithmetic* this
project teaches, only the scale and the weights:

| Aspect | This project | Production (EfficientNet/Swin/RETFound) |
|---|---|---|
| Weights | fixed, hand-seeded (untrained) | learned on 10⁴–10⁶ labelled images |
| Depth | 2 conv layers | dozens–hundreds; residual/attention blocks |
| Input | 32×32 synthetic, single image | 2048×2048 real, batched |
| Conv impl | hand-written tiled kernel | **cuDNN** (Winograd / implicit-GEMM), Tensor Cores, FP16 |
| Normalisation | none | batch/layer norm, augmentation |
| Output | single-task DR grade | multi-task (DR + glaucoma + AMD), calibrated uncertainty |
| Localisation | CAM stand-in | gradient Grad-CAM, attention rollout |
| Deployment | one `.exe` | **TensorRT** engines, dynamic batching, both eyes |

The most important honesty note: **fixed weights cannot diagnose**. A trained
model earns its accuracy from the *weights*, learned by backpropagation on
labelled data — which this project does not do. We teach the **forward-pass
mechanics and the GPU convolution pattern** that every such model shares. **Not
for clinical use.**

---

## 8. Further reading

- Gulshan et al., "Development and Validation of a Deep Learning Algorithm for
  Detection of Diabetic Retinopathy in Retinal Fundus Photographs," *JAMA* 2016.
- Tan & Le, "EfficientNet," ICML 2019 (compound scaling of CNN backbones).
- Zhou et al., "Foundation model for generalizable disease detection from retinal
  images" (**RETFound**), *Nature* 2023.
- Selvaraju et al., "Grad-CAM," ICCV 2017 (class-discriminative localisation).
- cuDNN documentation — how production convolutions are actually computed on GPU.
