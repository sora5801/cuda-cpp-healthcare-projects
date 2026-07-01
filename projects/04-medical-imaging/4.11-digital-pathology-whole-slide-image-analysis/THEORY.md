# THEORY — 4.11 Digital Pathology / Whole-Slide Image Analysis

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use._

> **Scope (CLAUDE.md §13).** This is a *reduced-scope teaching version* of a
> research-grade pipeline. We implement the **attention-MIL classification head**
> — the part CLAM uses to turn a bag of tile features into a slide-level call —
> over **pre-extracted** tile features. The upstream CNN/ViT encoder, tissue
> detection, and stain normalization are described in §7 but not implemented.

---

## 1. The science

A pathologist diagnoses cancer by looking at tissue under a microscope. A **whole-
slide image (WSI)** is that same glass slide digitized at up to 40× magnification:
a single image of **tens of billions of pixels** (0.5–5 GB compressed, stored as a
multi-resolution TIFF *pyramid*). No neural network can ingest a 100,000 × 100,000
image at once, and — crucially — for most slides we only have **one label for the
whole slide** ("this patient has cancer"), not a label for every pixel. We do not
know *where* the tumor is; we only know the slide *contains* one.

This is the setting of **weakly-supervised multiple-instance learning (MIL)**. The
standard recipe (CLAM, the catalog's baseline) is:

1. **Tissue detection** (Otsu threshold on a low-res level) to skip the white glass
   background — often 60–90% of the slide.
2. **Tiling**: cut the tissue into thousands of small patches (e.g. 256×256 at 20×).
   A slide yields `10⁴–10⁵` tiles; a cohort (TCGA, CAMELYON) yields *millions*.
3. **Feature extraction**: push each tile through a frozen pretrained encoder
   (ResNet-50, or a pathology foundation model like UNI) to get a `D`-dimensional
   **feature vector** per tile. This is where cuDNN/GPU time is spent in production.
4. **Attention MIL**: from the resulting **bag** of tile features, predict the slide
   label — while *learning which tiles matter* (the attention weights). The few
   tumor-bearing tiles should dominate; the stroma and normal tissue should not.

The output is both a **slide-level prediction** and an **attention heat map** that
tells a pathologist *which tiles the model found suspicious* — the interpretability
that makes MIL clinically appealing. **This project implements step 4.** Steps 1–3
are treated as an upstream black box: the committed sample already contains the
`N × D` features, as if the encoder had run.

## 2. The math

A slide is a **bag** of `N` tiles. Tile `i` has a feature vector `hᵢ ∈ ℝ^D`
(here `D = FEAT_DIM = 8`). The gated-attention MIL head (Ilse, Tomczak & Welling,
ICML 2018 — the head CLAM uses) has parameters:

- `V ∈ ℝ^{M×D}` — the attention "content" projection,
- `U ∈ ℝ^{M×D}` — the attention "gate" projection (`M = ATTN_HIDDEN = 4`),
- `w ∈ ℝ^{M}` — the attention combiner,
- `w_c ∈ ℝ^{D}`, `b_c ∈ ℝ` — the slide classifier.

**Per-tile attention logit** (a scalar relevance score for tile `i`):

```
eᵢ = wᵀ ( tanh(V hᵢ)  ⊙  σ(U hᵢ) )          σ(x) = 1/(1+e^{-x}),  ⊙ = elementwise
```

The `tanh` branch is the *content* (what the tile looks like); the `σ` branch is a
multiplicative **gate** (0…1) that can suppress or amplify each hidden unit per
tile. The gate is what makes gated attention more expressive than plain attention.

**Softmax over tiles** turns the `N` logits into a probability distribution — the
attention weights:

```
aᵢ = exp(eᵢ) / Σⱼ exp(eⱼ)        aᵢ ≥ 0,   Σᵢ aᵢ = 1
```

**Attention pooling** collapses the bag into a single slide embedding:

```
z = Σᵢ aᵢ hᵢ  ∈ ℝ^D           (an attention-weighted average tile feature)
```

**Slide classification** is a linear layer + logistic squashing:

```
s = w_cᵀ z + b_c              p = σ(s) ∈ (0,1)    (predicted tumor probability)
```

Inputs: the bag `{hᵢ}` and the (frozen) parameters. Outputs: the attention weights
`aᵢ` (the heat map), the embedding `z`, and the slide probability `p`.

## 3. The algorithm

```
for each tile i in parallel:          # STEP 1 — per-tile, independent
    eᵢ = wsi_attention_logit(hᵢ)       #   O(M·D) work per tile
m   = max_i eᵢ                         # STEP 2 — softmax (reduction)
aᵢ  = exp(eᵢ - m); Z = Σ aᵢ; aᵢ /= Z   #   subtract-max for stability
for each tile i in parallel:          # STEP 3 — weighted pool (reduction)
    for d: z[d] += aᵢ · hᵢ[d]          #   scatter-add into D accumulators
s   = w_c·z + b_c;  p = σ(s)           # STEP 4 — classify (O(D))
```

**Complexity.** Step 1 is `O(N·M·D)` total work with `O(M·D)=O(1)` *depth* per tile
(fully parallel across `N`). Steps 2 and 3 are `O(N·D)` work; a parallel reduction
has `O(log N)` depth. Step 4 is `O(D)`. So the serial cost is `O(N·(M·D))` and the
parallel *depth* is `O(M·D + log N)` — dominated by the tiny per-tile projection.
The arithmetic intensity is low (a few FLOPs per feature read), so at scale this is
**memory-bandwidth bound** on reading the `N·D` features — which is exactly why the
GPU (with ~10× the memory bandwidth of a CPU) wins as `N` grows.

## 4. The GPU mapping

**Thread-to-data mapping.** Both kernels use the canonical 1-D grid: tile
`i = blockIdx.x · blockDim.x + threadIdx.x`, one thread per tile, `256` threads per
block, `⌈N/256⌉` blocks. The ragged last block is guarded by `if (i >= N) return;`.

**Where each step runs.**

| Step | Where | Why |
|------|-------|-----|
| 1. logits | **kernel** `attention_logits_kernel` | embarrassingly parallel over tiles |
| 2. softmax | **host** | tiny exact reduction over `N` doubles; keeps stdout reproducible |
| 3. pool | **kernel** `attention_pool_kernel` | scatter-reduction via `atomicAdd` |
| 4. classify | **host** | `O(D)`; reuses the CPU code so both paths agree |

This "heavy parallel work on the device, small exact reduction on the host" split
is the same shape as flagship `11.09` (k-means: parallel assign kernel + host
centroid update).

**Memory hierarchy.**

- **Constant memory** holds the frozen model `AttnParams` (`cudaMemcpyToSymbol`).
  Every thread reads the *same* `V, U, w` — constant memory's per-SM broadcast
  cache serves a whole warp from one fetch, which is ideal for read-only,
  all-threads-identical data.
- **Global memory** holds the `N·D` tile features (read) and the `D` fixed-point
  accumulators (written via atomics). The feature read is the bandwidth bottleneck
  at scale; the access is coalesced because thread `i` reads the contiguous row `i`.
- **Registers** hold each tile's running logit and the `M·D` inner-loop temporaries.

```
   features[N][D]  (global, coalesced row per thread)      c_params (constant)
        │                                                        │
        ▼                                                        ▼
  ┌─────────────── attention_logits_kernel (1 thread / tile) ───────────────┐
  │ thread i:  eᵢ = w·(tanh(V·hᵢ) ⊙ σ(U·hᵢ))   →  logits[i]  (global)        │
  └─────────────────────────────────────────────────────────────────────────┘
        │  D2H copy logits
        ▼
     HOST: m=max eᵢ ; aᵢ = exp(eᵢ-m)/Σ ;   H2D copy attn[]
        │
        ▼
  ┌─────────────── attention_pool_kernel (1 thread / tile) ─────────────────┐
  │ thread i, for d:  atomicAdd(&fixed[d], quantize(aᵢ·hᵢ[d]))   (fixed pt)  │
  └─────────────────────────────────────────────────────────────────────────┘
        │  D2H copy fixed[D] ; dequantize
        ▼
     HOST: z ; s = w_c·z + b_c ; p = σ(s)
```

**Which library does what (no black boxes).** This teaching head uses **no** CUDA
library — the projection is a hand-written `M×D` dot product and the pool is a
hand-written atomic reduction, both small enough to read. In production the same
math is a `cuBLAS` GEMM: stacking the bag into `H ∈ ℝ^{N×D}`, the logits are
`(H Vᵀ)`-style matmuls and the pooled embedding is `aᵀ H` — a single
`cublasDgemv`/`cublasDgemm` call. The *feature-extraction* step (not here) is where
`cuDNN` runs the ResNet/ViT convolutions and `DALI` streams tile decode/augment.

## 5. Numerical considerations

- **Precision.** We use **FP64** throughout. The head is tiny, so double precision
  costs almost nothing and removes FP32 rounding as a variable while teaching the
  determinism idea. (A production model runs FP16/FP32 on tensor cores.)
- **Softmax stability.** We subtract `max_i eᵢ` before `exp` so every argument is
  `≤ 0` and cannot overflow. This shift cancels exactly in the ratio, so it does
  not change the result — only its conditioning.
- **The atomic-reduction determinism trap.** Step 3 has many tiles adding into the
  same `D` accumulators — a data race resolved by `atomicAdd`. If we summed
  **floating-point** contributions, the result would depend on the (hardware-
  nondeterministic) order in which atomics land, because float addition is **not
  associative**. The demo's stdout would then vary run to run. **Fix:** quantize
  each contribution `aᵢ·hᵢ[d]` to a **64-bit fixed-point integer** (`wsi_quantize`,
  scale `2³⁰`) and `atomicAdd` *those*. Integer addition **is** associative and
  commutative, so the sum is order-independent — deterministic **and** identical to
  the CPU's fixed-point sum. This is the same trick as flagships `5.01` (integer
  energy quanta) and `11.09` (fixed-point coordinate sums). Negative features are
  handled by storing signed values via two's-complement reinterpret (adds still
  commute).

## 6. How we verify correctness

`src/reference_cpu.cpp` provides an independent **serial** attention-MIL forward
pass. It shares the *per-tile* math with the GPU through the `__host__ __device__`
header `src/wsi.h`, and it uses the **same fixed-point pooling**, so the two
implementations should agree closely. `main.cu` runs both and checks three things:

- max |attention_cpu − attention_gpu|,
- max |embedding_cpu − embedding_gpu|,
- |probability_cpu − probability_gpu|, and that the top-attention tile index matches.

**Tolerance = 1e-9, and here is the honest reasoning** (PATTERNS.md §4). The
attention logits use `tanh` and `exp`; the **device** implementations of those
transcendentals differ from the **host** libm by ~1 ULP (~1e-16). That tiny
difference shows up in the attention weights (`max attn diff ≈ 1e-16` on the
sample). But the **pooling is fixed-point**: a 1e-16 perturbation of `aᵢ` almost
never crosses a `2⁻³⁰` quantization boundary, so the integer accumulators — and
therefore the embedding and the final probability — come out **bit-identical**
(`embed diff = 0`, `prob diff = 0`). We verify to **1e-9** (far tighter than
anything that could change the clinical-style readout) and report the raw diffs on
stderr rather than claiming bit-exactness. Two independent implementations (a
serial CPU loop and a parallel GPU grid) agreeing to that tolerance is strong
evidence the computation is correct.

**Interpretability check (validates the science, not just CPU==GPU).** The sample
plants 6 tumor tiles at known indices; the demo confirms attention concentrates on
exactly those tiles and the slide is called `TUMOR`, while a `--tumor-frac 0` slide
yields flat attention and a `benign` call. That the *right tiles* light up is the
real correctness signal for an attention model.

## 7. Where this sits in the real world

Production WSI classification (CLAM, and its many descendants) differs from this
teaching head in scope, not in the core idea:

- **Feature extraction.** Real pipelines spend almost all their GPU time in step 3:
  running a ResNet-50 or a pathology **foundation model** (UNI, a ViT pretrained
  with DINO/MAE self-supervision on 100k+ slides) over every tile via **cuDNN**,
  with **DALI** streaming tile decode/normalize/augment to keep the GPU fed. We
  skip this entirely and consume the features.
- **Tissue detection & stain normalization.** Otsu thresholding removes glass
  background; **Macenko/Vahadane** stain normalization corrects scanner/lab color
  variation so features transfer across sites. Not implemented here.
- **Learned, not frozen.** The attention head and classifier are **trained** by
  backpropagation from slide labels (with a clustering-based instance loss in CLAM
  for extra supervision). Our weights are hand-set so the demo is reproducible.
- **Scale & engineering.** Real bags are `10⁴–10⁵` tiles; cohorts are millions.
  The logits/pool become **cuBLAS** GEMMs, features stream with **pinned memory**,
  and extraction is sharded across GPUs with `torch.multiprocessing`.
- **Beyond binary.** Multi-class CLAM, survival prediction (Cox loss), and
  **spatial-transcriptomics** co-registration (aligning histology with per-spot
  gene expression) are second-order workloads named in the catalog.

**None of this is diagnostic.** Even the full pipeline is a decision-support
research tool; our reduced version on synthetic data is purely didactic.

---

## References

- **Ilse, Tomczak & Welling (2018), "Attention-based Deep MIL", ICML** — the gated-
  attention pooling this project implements. The single most important paper here.
- **Lu et al. (2021), "Data-efficient and weakly supervised... (CLAM)", Nature BME**
  — the standard WSI-MIL baseline; our head is CLAM's classification head.
  <https://github.com/mahmoodlab/CLAM>
- **Chen et al. (2024), "UNI", Nature Medicine** — a pathology foundation-model
  encoder (the "feature extractor" we treat as upstream).
  <https://github.com/mahmoodlab/UNI>
- **OpenSlide** <https://openslide.org/> — reads the WSI pyramid formats; study its
  API to understand tiling and multi-resolution access.
- **HistomicsTK** <https://github.com/DigitalSlideArchive/HistomicsTK> — GPU WSI
  toolkit (stain normalization, tissue detection) for the steps we omit.
- **Macenko et al. (2009)** — the stain-normalization method named in the catalog.
