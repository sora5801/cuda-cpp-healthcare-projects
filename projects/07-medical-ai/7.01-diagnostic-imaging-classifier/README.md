# 7.1 — Diagnostic Imaging Classifier

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Medical%20AI%20%26%20Clinical%20Deep%20Learning-lightgrey)

> **🟢 Beginner · Established** — Domain 7: Medical AI & Clinical Deep Learning · Catalog ID `7.1`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._ This is a
> **reduced-scope teaching version** (CLAUDE.md §13): a hand-written CNN
> **forward pass** with fixed synthetic weights on synthetic image patches. It
> teaches the compute-dominant math and its GPU mapping — it does not train a
> model and must never be used for any medical decision.

## Summary

This project runs the classic minimal image classifier — **conv → ReLU →
max-pool → dense → softmax** — on a small batch of grayscale "image patches" and
predicts one of two classes: `normal` tissue or a `lesion` (a bright blob
standing in for a nodule/mass). We implement the whole forward pass twice: once
as an obvious serial CPU reference, and once as hand-written CUDA kernels that
assign **one GPU thread per output pixel**. The two agree bit-for-bit, so you can
read the parallel code with confidence that it computes exactly the same thing.
The point is to see, with nothing hidden, why a convolutional network is a
data-parallel workload and how it lands on GPU threads, constant memory, and a
gather access pattern.

## What this computes & why the GPU helps

Real diagnostic-imaging models (the catalog deep-dive below) train convolutional
and transformer networks to classify pathologies from CT/MRI/X-ray/ultrasound.
The **compute-dominant operation is convolution**: for every output pixel of
every feature map of every image you compute a small `K×K` dot product. A single
`512×512` slice already has hundreds of thousands of pixels; a batch of them
across dozens of filters is tens of millions of independent dot products.

> Trains convolutional and transformer-based networks to classify pathologies (malignancy, disease grade, anatomical anomaly) from 2D/3D medical images — CT, MRI, X-ray, ultrasound. GPUs provide the tensor-parallel matrix multiply needed to process high-resolution volumetric input in minibatches; a single 512×512 CT slice stack can reach tens of millions of pixels. Backbone convolutions (3D U-Net, ResNet-50, EfficientNet, ViT-B) are the compute-dominant operation, mapping directly onto CUDA tensor cores. Mixed-precision FP16/BF16 training via cuDNN doubles effective throughput versus FP32 while preserving classification accuracy. Inference on edge devices is further accelerated with TensorRT INT8 quantisation.

**The parallel bottleneck:** the **convolution layer**. Each of
`batch × filters × out_h × out_w` output pixels is an independent `K×K`
multiply-accumulate (a "gather" over a local window). We give each output pixel
its own GPU thread, so all of them are computed at once — exactly the parallelism
cuDNN/tensor cores exploit in production, written here by hand so it is not a
black box.

## The algorithm in brief

- **Conv2D (valid padding):** `NUM_F` filters of size `K×K` slide over the image;
  each produces a feature map. Implemented in the shared `conv_pixel()`.
- **Bias + ReLU:** add a per-filter bias, then `max(0, x)` — the nonlinearity that
  gives the network expressive power.
- **2×2 max-pool (stride 2):** keep the strongest response in each 2×2 block →
  translation tolerance and a 4× spatial shrink (`pool_pixel()`).
- **Flatten + dense (fully connected):** one dot product per class over the pooled
  features → two logits (`dense_logit()`).
- **Softmax + argmax:** turn logits into `P(lesion)` and a predicted class.

The catalog also lists the full-scale toolbox (below); this teaching version
implements the core forward path. See [THEORY.md](THEORY.md) for the full
science → math → algorithm → GPU-mapping derivation.

> Catalog key algorithms: 3D convolutional neural networks (ResNet-3D, DenseNet), Vision Transformers (ViT, Swin-T), EfficientNet, data augmentation with random affine/elastic transforms, AUC-optimised losses, Grad-CAM explainability, TTA (test-time augmentation) ensembling.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/diagnostic-imaging-classifier.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/diagnostic-imaging-classifier.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\diagnostic-imaging-classifier.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/imaging_sample.txt`, prints the
per-image prediction table and batch accuracy, shows the GPU-vs-CPU agreement
check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/imaging_sample.txt` — a tiny, **synthetic**
  input (4 image patches + the fixed model weights) so the demo runs with zero
  downloads. Regenerate/edit it with `python scripts/make_synthetic.py`.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print instructions and
  links for the real datasets (all require registration and forbid casual
  redistribution — the scripts never bypass credentials).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: MIMIC-CXR — 227,827 labelled chest X-ray studies (https://physionet.org/content/mimic-cxr/) · CheXpert — 224,316 chest X-rays, 14 pathology labels (https://stanfordmlgroup.github.io/competitions/chexpert/) · LIDC-IDRI — 1,018 CT lung-nodule cases (https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) · The Cancer Imaging Archive / TCIA (https://www.cancerimagingarchive.net/).

## Expected output

Success looks like `demo/expected_output.txt`: two lesion patches predicted
`lesion` with high `P(lesion)`, two normal patches predicted `normal`, 4/4
accuracy, and `RESULT: PASS`. The program computes the forward pass on both the
**GPU** (`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`) and
asserts the class logits agree **exactly** (tolerance `0`) — because both call the
identical `__host__ __device__` math. That exact agreement is the correctness
guarantee; the `[verify]` line on stderr shows `max |logit_cpu - logit_gpu| = 0`.

## Code tour

Read in this order:

1. [`src/reference_cpu.h`](src/reference_cpu.h) — the model geometry, the
   `Weights`/`Dataset` structs, and the **shared `__host__ __device__` math**
   (`conv_pixel`, `pool_pixel`, `dense_logit`, `softmax_pos1`). Start here.
2. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
4. [`src/kernels.cu`](src/kernels.cu) — the two kernels (conv+pool, dense) and the
   host wrapper; constant memory for weights.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline
   and the synthetic-data generator.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **MONAI** (https://github.com/Project-MONAI/MONAI) — PyTorch-native medical
  imaging framework with C++/CUDA extensions for resampling/transforms. Study how
  a real training/inference pipeline is structured.
- **TorchXRayVision** (https://github.com/mlmed/torchxrayvision) — pretrained chest
  X-ray models and CheXpert/MIMIC-CXR loaders; a good "what real labels look like".
- **nnU-Net** (https://github.com/MIC-DKFZ/nnUNet) — auto-configuring baseline that
  wins most medical-imaging benchmarks; learn its preprocessing/augmentation choices.
- **TotalSegmentator** (https://github.com/wasserth/TotalSegmentator) — 104-structure
  CT segmentation built on nnU-Net.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Independent per-output-pixel gather + constant-memory weights** (PATTERNS.md
§1, the pattern exemplified by flagship `7.10` for convolution and `1.12` for
constant-memory broadcast). One thread computes one pooled feature; the small,
never-changing network weights live in `__constant__` memory so a warp reading
the same filter tap is served in a single broadcast. The verification uses the
**shared `__host__ __device__` core** (PATTERNS.md §2) for exact CPU/GPU parity.

> Catalog CUDA pattern (full scale): cuDNN for convolution kernels, NCCL for multi-GPU data-parallel training, TensorRT for deployment; minibatch data parallelism with NCCL all-reduce, optional model parallelism for 3D volumes that exceed single-GPU VRAM.

## Exercises

1. **Bigger and heavier.** Raise `IMG_H/IMG_W`, `NUM_F`, and the batch size in
   `reference_cpu.h` (and `make_synthetic.py`), rebuild, and watch the GPU-vs-CPU
   timing gap on stderr grow as compute starts to dominate launch overhead.
2. **Shared-memory tiling.** The current conv kernel re-reads overlapping input
   windows from global memory. Port the tiling trick from flagship `7.10` (stage a
   block of the image + halo into shared memory) and confirm the result is unchanged.
3. **Add a second conv layer.** Insert another conv→ReLU→pool stage before the
   dense layer (add a `conv_pixel`-style helper over feature maps). How does the
   receptive field change?
4. **Grad-CAM (explainability).** For a lesion prediction, back out which spatial
   locations contributed most to class 1 by weighting the filter-0 feature map by
   its dense weights — a one-layer stand-in for the catalog's Grad-CAM.
5. **Batched dense as GEMM.** Replace `dense_kernel` with a cuBLAS `Sgemm`
   (`features [n×FLAT] · Wᵀ [FLAT×NUM_CLS]`) and compare — this is how real
   frameworks do the fully-connected layer (see flagship `3.11` for GEMM).

## Limitations & honesty

- **Reduced scope:** this is **inference only** with **fixed, hand-designed
  weights**. There is no training, no backprop, no data augmentation, no
  transformers, no cuDNN/tensor cores, no mixed precision, and no TensorRT — all of
  which the catalog entry names. THEORY.md §"Where this sits in the real world"
  maps the gap.
- **Synthetic data:** the "images" are generated blobs/gradients labeled
  `synthetic`; they are **not** medical images and carry no clinical meaning.
- **Tiny problem:** the batch is 4 images so the demo is instant and the numbers
  are hand-checkable; on this size the GPU is *slower* than the CPU (launch/copy
  overhead) — the timing is a teaching artifact, never a benchmark claim.
- **Not for clinical use.** Nothing here may inform a diagnosis or treatment.
