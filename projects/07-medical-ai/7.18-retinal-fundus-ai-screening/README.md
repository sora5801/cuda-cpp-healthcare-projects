# 7.18 — Retinal Fundus AI Screening

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Medical%20AI%20%26%20Clinical%20Deep%20Learning-lightgrey)

> **🟢 Beginner · Established** — Domain 7: Medical AI & Clinical Deep Learning · Catalog ID `7.18`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Automated screening for diabetic retinopathy (DR) grades colour photographs of
the back of the eye (the *fundus*) on a 0–4 severity scale. The workhorse is a
**convolutional neural network (CNN)**: stacked convolution + activation +
pooling layers that turn raw pixels into a small feature vector, followed by a
linear classifier. This project implements that CNN **forward pass (inference)**
on the GPU as a heavily-commented teaching model — the same skeleton as
EfficientNet/Swin/RETFound, but small and with **fixed (untrained) weights** so
it runs deterministically with no training run, no ML framework, and no
downloads. The headline GPU lesson is the **shared-memory tiled 2-D
convolution**, the 2-D analog of flagship 7.10.

## What this computes & why the GPU helps

Classifies diabetic retinopathy (and, in production, glaucoma and AMD) from
colour fundus photographs. High-resolution fundus images (typically 2048×2048)
require significant GPU memory for batch processing; ResNet, EfficientNet, and
Swin-Transformer backbones fine-tuned on annotated fundus datasets are the
standard approach. GPU tensor cores accelerate the backbone convolutions in
batch; simultaneous inference across both eyes and multiple pathologies
(multi-task heads) doubles effective throughput. Real-world screening pipelines
process millions of images annually, making GPU throughput a primary operational
concern.

**The parallel bottleneck:** the **convolution layers**. Every output pixel of
every feature map is an independent `K×K×C_in` multiply-accumulate over a local
neighbourhood — millions of them per image, all independent. We give each output
pixel its own GPU thread and stage each input tile in **shared memory** once (a
halo tile) so neighbouring threads reuse loads instead of re-reading global
memory `K×K` times. Pooling, global-average-pooling (a block reduction), and the
tiny classifier round out the pipeline.

## The algorithm in brief

- **2-D convolution + ReLU** (`conv_relu_tiled`) — the shared-memory tiling kernel.
- **2×2 max-pool** — halve spatial size, keep the strongest activation.
- Two conv→ReLU→pool blocks (3→6→12 feature maps), then **global average pool**
  → a 12-D embedding.
- **Fully-connected classifier** → 5 logits → **softmax** → argmax = DR grade.
- **Grad-CAM-style class-activation map** for lesion localisation (catalog
  "Grad-CAM").

Production systems add: learned weights (trained on EyePACS/APTOS), deeper
backbones (EfficientNet-B4/B5, Swin Transformer), self-supervised pretraining
(RETFound), multi-task heads, and uncertainty calibration for referral. See
[THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation and §7 for what a *trained* model changes.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/retinal-fundus-ai-screening.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/retinal-fundus-ai-screening.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\retinal-fundus-ai-screening.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/fundus_sample.txt`, prints the
predicted grade + probabilities + Grad-CAM peak, shows the GPU-vs-CPU agreement
check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/fundus_sample.txt` — one tiny, clearly
  **synthetic** 3×32×32 colour "fundus-like" image so the demo runs offline with
  zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print instructions and
  links (all real datasets are account-gated; the scripts never bypass a login).
  `scripts/make_synthetic.py` regenerates the synthetic sample at any size.
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: EyePACS — 88,000 labelled fundus images, 5-grade DR
severity (Kaggle, verify URL); APTOS 2019 — 3,662 fundus images (Kaggle, verify
URL); DRIVE / STARE — vessel segmentation (verify URL); UK Biobank Retinal
Imaging — 68k fundus images with linked health records
(<https://www.ukbiobank.ac.uk/>).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt):

```
7.18 -- Retinal Fundus AI Screening
[teaching CNN inference: conv->relu->pool x2 -> GAP -> FC -> softmax]
image: 32x32 RGB  (channel-major, normalized [0,1])
predicted DR grade: 2 moderate
class probabilities: 0.211474 0.201075 0.233411 0.154383 0.199657
Grad-CAM 8x8 peak = 0.225891 at (row=4,col=2)
RESULT: PASS (GPU matches CPU within tol=1.0e-03; same grade)
```

The program runs the forward pass on both the **GPU** (`src/kernels.cu`) and a
**CPU reference** (`src/reference_cpu.cpp`) using the *same* per-pixel math
(`src/cnn_core.h`), and asserts the logits/probabilities/CAM agree within `1e-3`
**and** that both predict the same grade — that agreement is the correctness
guarantee. (The actual error is ~`1e-8`; the tolerance covers float-summation
order in the average-pool reduction — see THEORY §5.)

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the image, runs CPU + GPU, verifies, reports.
2. [`src/cnn_core.h`](src/cnn_core.h) — the shared `__host__ __device__` per-pixel
   math (conv MAC, ReLU, max-pool) that guarantees CPU==GPU parity.
3. [`src/reference_cpu.h`](src/reference_cpu.h) / [`.cpp`](src/reference_cpu.cpp) —
   the image/model containers and the trusted serial forward pass.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the tiling idea.
5. [`src/kernels.cu`](src/kernels.cu) — the tiled conv, pool, GAP, FC, and CAM kernels.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, I/O helpers.

## Prior art & further reading

- **EfficientNet / EfficientDet** (<https://github.com/google/automl/tree/master/efficientnet>)
  — strong fundus baselines; study the compound-scaling idea behind the depth we
  omit.
- **MONAI** (<https://github.com/Project-MONAI/MONAI>) — production medical-imaging
  pipelines; see its fundus classification transforms and training loops.
- **RETFound** (<https://github.com/rmaphoh/RETFound_MAE>) — a masked-autoencoder
  foundation model pretrained on 1.6M fundus images; the self-supervised
  pretraining we mention in "further work".
- **DeepDR Plus** — end-to-end DR screening system (verify URL).

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Shared-memory tiling** for the convolution layers (the 2-D version of the 1-D
tiling in flagship 7.10): a block loads a haloed input tile into on-chip shared
memory once, then every thread reads its `K×K` window from there. Plus a **block
reduction** for global average pooling and small element-wise kernels for
pooling / classifier / CAM. Production stacks use **cuDNN** for the convolutions
and **TensorRT** for deployment; this project hand-writes the conv so the
mechanism is not a black box (CLAUDE.md §6.1.6).

## Exercises

1. **Bigger, faster.** Regenerate a 128×128 sample (`make_synthetic.py --size
   128`) and watch the stderr timing: the GPU's edge over the CPU grows as the
   image (and thus the conv work) grows. At what size does the GPU overtake the
   CPU on your card?
2. **Tune the tile.** Change `TILE` in `kernels.cuh` (e.g. 8, 16, 32) and compare
   kernel time. Note the shared-memory-per-block limit and occupancy trade-off.
3. **Add a layer.** Extend the model to a third conv→ReLU→pool block (update
   `cnn_core.h` shapes, `make_fixed_model`, both forward paths). Does CPU==GPU
   still hold?
4. **True Grad-CAM.** Our CAM uses the FC weights directly; implement gradient-
   based Grad-CAM (backprop the winning logit to the last conv feature maps) and
   compare the heatmaps.
5. **Load a real image.** Convert a real fundus photo to the text format
   (`data/README.md`), run it, and confirm the (untrained) pipeline still
   produces a valid probability vector — a reminder that architecture ≠ accuracy.

## Limitations & honesty

- **Not for clinical use.** The weights are **fixed and untrained**, so the
  predicted grade is meaningless as a diagnosis — this demonstrates the CNN
  *inference pipeline*, not diagnostic performance.
- **Synthetic data.** The committed image is hand-drawn and clearly synthetic; it
  is not a real retina.
- **Reduced scope.** Two shallow conv layers vs. the dozens in EfficientNet/Swin;
  no batch norm, no learned weights, no multi-task heads, single image (no batch),
  fixed small resolution. THEORY §7 lists exactly what production changes.
- **Teaching kernels.** The conv kernel favours clarity over peak throughput (no
  `im2col`+GEMM, no Tensor Cores, no cuDNN); it is fast enough to teach the
  tiling idea and to verify against the CPU, not to set records.
