# THEORY — 4.7 Medical Image Segmentation (Deep Learning)

> The deep didactic explanation (the "why"). Written for a sharp student who
> knows C++ but is new to CUDA and new to this domain. See [README.md](README.md)
> for the quick tour and build steps.
>
> _Educational only — not for clinical use. This is a **reduced-scope teaching
> version**: a fixed-weight 3D-convolution segmentation head, not a trained
> nnU-Net. §7 explains exactly what the full pipeline adds._

---

## 1. The science

**The problem.** *Segmentation* assigns a class label to every voxel of a 3D
medical scan: "this voxel is liver, that one is tumor, this one is background."
Unlike *classification* (one label per image) or *detection* (a bounding box),
segmentation produces a dense, per-voxel mask — the shape and volume of a
structure, which is what clinicians and downstream tools actually need
(tumor-volume tracking, organ dosimetry for radiotherapy, surgical planning,
quantitative biomarkers).

A CT or MRI study is a stack of 2D slices, i.e. a 3D array of intensities
`I(z, y, x)`. For CT the intensity is in Hounsfield units (air ≈ −1000, water ≈
0, bone ≈ +1000); for MRI it is a relative signal. The job is to learn a function
`f: I → L` mapping the intensity volume to a label volume `L(z, y, x) ∈ {0..K−1}`.

**Why deep learning.** Hand-written rules ("threshold at 300 HU for bone") work
for high-contrast structures but fail on soft tissue, where organs have
overlapping intensities and are distinguished only by **context and shape**.
Convolutional neural networks (CNNs) learn that context: stacked 3D convolutions
build up features from edges → textures → organ parts → organs. The dominant
architecture is the **3D U-Net** — an encoder that downsamples to capture
context, a decoder that upsamples back to voxel resolution, and skip connections
that restore fine detail. `nnU-Net` auto-configures a U-Net to a dataset and is a
state-of-the-art baseline; `TotalSegmentator` segments 117 whole-body structures.

**What this teaching project does.** Training a 3D U-Net needs thousands of
labelled volumes and a GPU-week — out of scope for a self-contained study repo.
Instead we isolate the **single primitive those networks are built from and spend
~90% of their FLOPs on: the 3D convolution** — and run a tiny *fixed-weight*
2-layer fully-convolutional head that segments one structure (a bright "lesion")
from a synthetic volume. The weights are hand-set to an interpretable rule
(denoise, then threshold the local mean intensity), so the math is transparent
and the result is reproducible to the bit. The GPU mechanics are identical to a
real conv layer; only the weights are given instead of learned.

## 2. The math

A **3D convolution layer** maps an input tensor with `Cin` channels to an output
tensor with `Cout` channels. For output channel `o` and voxel `(z, y, x)`:

```
        Cin−1   R    R    R
z_o(p) = Σ      Σ    Σ    Σ   W[o,c,dz,dy,dx] · X[c, z+dz, y+dy, x+dx]  +  b[o]
        c=0   dz=−R dy=−R dx=−R
```

- `X[c, ·]` — input feature map `c` (channel-major, then row-major `(z,y,x)`).
- `W[o,c,·]` — the learned `(2R+1)³` weight stencil ("filter") for output `o`,
  input `c`. Here `R = 1`, so each filter is `3×3×3 = 27` taps.
- `b[o]` — the per-output-channel bias.
- Border voxels use **zero padding** ("same" convolution): taps that reach
  outside the volume contribute 0, so output and input have the same dimensions.

A layer applies a pointwise non-linearity afterwards; we use the standard
**ReLU**, `σ(t) = max(0, t)`, between the two conv layers.

**Our two-layer head** (shapes for input `X[1, D, H, W]`):

```
layer 1 :  H = ReLU( conv3x3x3( X ; W1[Chid,1,27], b1 ) )     -> H[Chid, D,H,W]
layer 2 :  Z =        conv3x3x3( H ; W2[K,Chid,27], b2 )       -> Z[K,    D,H,W]
argmax  :  L(z,y,x) = argmax_k Z[k, z,y,x]                     -> L in {0..K−1}
```

with `Chid = 2` hidden channels and `K = 2` classes (0 = background, 1 = lesion).

**The fixed weights** (see `make_segnet` in `reference_cpu.cpp`) encode an
explicit, readable detector:

- Layer 1, channel 0 = a normalized **3×3×3 Gaussian** (σ = 0.9): a low-pass
  smoother that removes the per-voxel noise.
- Layer 1, channel 1 = **identity** (only the center tap is 1): passes the raw
  intensity through (unused by the current rule; room for experiments).
- Layer 2, class 1 (lesion) = a uniform **box average** (`1/27` on every tap of
  the smoothed channel) with bias `−τ`, `τ = 0.46`. So its logit is
  `mean₂₇(smoothed) − τ` — positive exactly where the local mean intensity
  exceeds the lesion threshold.
- Layer 2, class 0 (background) = all-zero filter, bias 0 → logit `≡ 0`.

`argmax` then labels a voxel "lesion" iff `mean₂₇(smoothed) > τ`. This is
**denoise-then-threshold expressed as two conv layers** — the simplest non-trivial
CNN segmenter, and a genuine instance of the architecture a trained U-Net uses.

**Accuracy metric — Dice.** Against the ground-truth mask `G` we report the Dice
similarity coefficient of the predicted mask `P`:

```
Dice(P, G) = 2 |P ∩ G| / (|P| + |G|)   ∈ [0, 1],   1 = perfect overlap.
```

Dice is the standard segmentation score; it is computed from **integer voxel
counts**, so it is exact and machine-independent.

## 3. The algorithm

```
load volume X[1,D,H,W] (+ ground-truth mask G)        # data/sample
build fixed weights W1,b1,W2,b2                        # make_segnet (no training)
# --- forward pass ---
for each voxel (z,y,x):                                # layer 1
    for co in 0..Chid-1:  H[co,z,y,x] = ReLU( conv3x3x3(X, W1[co], b1[co]) )
for each voxel (z,y,x):                                # layer 2 + argmax
    for k in 0..K-1:      Z[k] = conv3x3x3(H, W2[k], b2[k])
    L[z,y,x] = argmax_k Z[k]
# --- score ---
Dice(L, G)
```

**Complexity.** Each output voxel of a layer costs `Cin · 27` fused multiply-adds.
Total work for the head is

```
O( D·H·W · (Chid·27  +  K·Chid·27) )  =  O(N · 27 · (Chid + K·Chid))
```

i.e. **linear in the number of voxels** `N = D·H·W` (the per-voxel constant is the
filter size × channels). A real 3D U-Net is the same per-layer cost summed over
dozens of layers with `Cin, Cout` up to 320 — hundreds of GFLOPs for one `512³`
volume. The **arithmetic intensity** is modest: each input voxel is reused by `27`
neighbouring output voxels, so a naive kernel re-reads global memory ~27× (a real
conv kernel tiles the input into shared memory to fix this; see §4 and the
Exercises). The access pattern is a dense, regular **3D stencil** — perfectly
parallel, no data dependence between output voxels.

## 4. The GPU mapping

**Pattern:** *3D stencil / gather, one thread per output voxel* (PATTERNS.md §1,
the same family as `4.01` CT backprojection and `14.02` reaction-diffusion).

**Thread-to-data mapping.** We flatten the volume to `N = D·H·W` voxels and launch
a 1-D grid of `SEG_BLOCK = 256`-thread blocks. Thread

```
i = blockIdx.x · blockDim.x + threadIdx.x
```

owns output voxel `i`; it decodes `(z, y, x)` from `i` (`x = i mod W`,
`y = (i/W) mod H`, `z = i/(H·W)`), gathers its `3×3×3` neighbourhood, and reduces
it against the filter. `kernels.cu` runs two kernels back-to-back:

- `seg_layer1_kernel` — each thread computes all `Chid` hidden channels for its
  voxel (a tiny unrolled loop) and applies ReLU.
- `seg_layer2_kernel` — each thread computes the `K` class logits, takes the
  argmax (strict `>` so ties keep the lower class index), and writes the label +
  the lesion logit.

```
   grid  : ceil(N / 256) blocks                     N = D*H*W voxels
   block : 256 threads (8 warps; multiple of 32)
                                                     thread i -> voxel (z,y,x)
   global memory                 constant memory          per thread
   ┌───────────────┐             ┌───────────┐            ┌──────────────┐
   │ X / H feature │ gather 3x3x3 │  filter   │  broadcast │ acc = Σ w·x  │
   │     maps      │────────────► │  weights  │──────────► │   + bias     │
   └───────────────┘             └───────────┘            └──────────────┘
                                                          write H / label
```

**Memory hierarchy and why.**

- The **weights** are tiny (a few hundred floats), read by every thread, and
  constant during a launch → they live in `__constant__` memory. When all threads
  in a warp read the same tap, the constant cache broadcasts it in **one**
  transaction. This is the textbook use of constant memory for shared read-only
  filter coefficients.
- The **feature maps** live in global memory; each thread issues `Cin·27` global
  loads for its neighbourhood. Neighbouring threads read overlapping windows, so
  the L1/L2 cache absorbs much of the reuse here. (A production kernel — or cuDNN
  — additionally **tiles** a block of the volume + a halo into `__shared__` memory
  so each input voxel is read from global memory once; that is the same
  shared-memory-tiling lesson as flagship `7.10`, left as an Exercise.)
- Per-thread accumulators (`acc`, the argmax state) live in **registers**.

**Occupancy / bandwidth.** 256 threads/block gives the scheduler 8 warps to hide
the global-load latency of the gather; with only a handful of registers and no
shared memory the kernel is limited by occupancy only at very large channel
counts. On this toy `12×16×16` volume the kernels are **launch-bound** (the work
is microseconds), so the reported GPU time is dominated by launch overhead — an
honest teaching caveat (§"Timing" in PATTERNS.md §7). The GPU's edge appears at
real `512³` volumes with hundreds of channels, where cuDNN turns this into a
Tensor-Core GEMM (via im2col or implicit-GEMM/Winograd).

**Which library does what (no black boxes).** Production code calls
**cuDNN** `cudnnConvolutionForward`, which internally reshapes the convolution
into a matrix multiply (im2col → GEMM on Tensor Cores) or uses a Winograd/FFT
algorithm. We **hand-roll** the conv here so the 27-tap dot product is visible;
writing the cuDNN-equivalent by hand would mean building the im2col buffer,
calling cuBLAS GEMM, and adding bias+activation — the subject of a later project.

## 5. Numerical considerations

- **Precision: FP32.** Deep-learning inference runs in single (or half)
  precision; we use FP32 everywhere. The per-voxel convolution sums only `Cin·27 ≤
  54` terms of similar magnitude, so FP32 is plenty — no catastrophic
  cancellation.
- **Determinism of the label map.** Every thread computes its own voxel's logits
  from a **fixed-order** loop (the shared `conv3x3x3_at` walks channels then taps
  in a fixed sequence) — there is **no cross-thread reduction**, no `atomicAdd`,
  no order dependence. The `argmax` uses strict `>` so ties deterministically pick
  the lower class index, exactly as the CPU does. Hence the integer label map is
  **bit-identical** between runs and between CPU and GPU → `stdout` is
  reproducible (PATTERNS.md §3).
- **FMA and the float logits.** The GPU contracts `a*b + c` into a single
  fused-multiply-add (FMA) with one rounding; the host compiler may use two
  roundings. So the continuous logits differ by ~`1e-7`–`1e-6`. This essentially
  never flips an `argmax` here (the lesion boundary is sharp), so labels still
  match exactly; we nonetheless verify the logits to a documented `1e-3` tolerance
  (PATTERNS.md §4) rather than pretend they are bit-identical.
- **Race conditions: none.** Output voxels are independent; each is written by
  exactly one thread. Layer 2 reads layer 1's output only after `seg_layer1`
  finishes (kernels are serialized in the default stream), so there is no
  read-before-write hazard.

## 6. How we verify correctness

Three independent checks (`main.cu` step 4–5):

1. **GPU label map == CPU label map, exactly.** The CPU reference (`segment_cpu`)
   and the GPU share the *same* `conv3x3x3_at` / `relu` core (the
   `__host__ __device__` idiom, PATTERNS.md §2), so the integer masks must agree
   for **every** voxel. `label mismatches` must be 0 — a hard gate. An independent
   serial implementation agreeing voxel-for-voxel with the parallel one is strong
   evidence the parallelization (index math, boundary handling, argmax tie-break)
   is correct.
2. **Lesion logits agree within `1e-3`.** The continuous scores match to ~`1e-7`
   in practice (well under tolerance); the small slack accounts for FMA
   differences (§5).
3. **Dice against known ground truth ≈ 0.96.** Because the synthetic lesion's
   location is *known*, we score the prediction against it. This validates the
   **science** (the head actually finds the lesion), not merely CPU==GPU
   agreement — the second, stronger check PATTERNS.md §4 recommends. Edge cases
   covered by the boundary logic: lesion voxels touching the volume border (zero
   padding), and the noisy soft edge of the sphere (the smoother + box average
   reject it cleanly).

## 7. Where this sits in the real world

This project is a deliberately reduced-scope teaching version. A production
segmentation pipeline differs in every dimension **except the core 3D convolution**:

| Aspect | This teaching head | Real nnU-Net / MONAI / TotalSegmentator |
|---|---|---|
| Weights | hand-set, fixed | **learned** from thousands of labelled volumes |
| Depth | 2 conv layers | encoder–decoder U-Net, dozens of layers, skip connections |
| Channels | 1 → 2 → 2 | up to 320 feature channels |
| Receptive field | 5³ voxels | whole-organ context via downsampling |
| Classes | 2 (bg/lesion) | up to 117 anatomical structures |
| Conv engine | hand-rolled 27-tap dot product | **cuDNN** im2col/Winograd GEMM on Tensor Cores |
| Precision | FP32 | mixed FP16/BF16 (Tensor Cores) |
| Large volumes | whole tiny volume in one go | **sliding-window patch** inference, stitched |
| Memory | a few KB | ~16 GB; CUDA Unified Memory for big volumes |
| Training | none | data augmentation (DALI/MONAI), DDP + NCCL multi-GPU |
| Post-proc | none | CRF / largest-connected-component cleanup |

What carries over exactly: the per-voxel 3D-conv arithmetic, the one-thread-per-
voxel mapping, constant-memory weights, zero-padded "same" convolution, the
ReLU/argmax, and the Dice metric. Swap the fixed weights for trained ones and add
the encoder/decoder structure and you have the real thing — the GPU primitive you
profiled here is the one nnU-Net spends its time in.

---

## References

- **Ronneberger et al., "U-Net" (2015)** and **Çiçek et al., "3D U-Net" (2016)** —
  the encoder-decoder-with-skip-connections architecture this head is a slice of.
- **Isensee et al., "nnU-Net" (Nature Methods, 2021)**
  (https://github.com/MIC-DKFZ/nnUNet) — self-configuring U-Net; the universal
  baseline. Study how it picks patch/batch size and topology from a dataset
  fingerprint.
- **Wasserthal et al., "TotalSegmentator" (2023)**
  (https://github.com/wasserth/TotalSegmentator) — 117-structure whole-body CT;
  the production target this teaching project gestures at.
- **MONAI** (https://github.com/Project-MONAI/MONAI) — PyTorch medical-imaging
  framework; read its GPU-resident transforms and network zoo.
- **Swin-UNETR** (https://github.com/Project-MONAI/research-contributions) —
  transformer encoder for 3D segmentation; shows the attention alternative to
  pure convolution (and why it is VRAM-hungry).
- **cuDNN Developer Guide** — how `cudnnConvolutionForward` maps a convolution to
  im2col/implicit-GEMM/Winograd on Tensor Cores; the "what we'd call instead of
  hand-rolling" reference for §4.
