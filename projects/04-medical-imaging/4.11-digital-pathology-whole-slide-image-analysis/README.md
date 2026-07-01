# 4.11 — Digital Pathology / Whole-Slide Image Analysis

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.11`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

> **Scope note (teaching version, CLAUDE.md §13).** The full project is a deep-
> learning pipeline (per-tile CNN/ViT feature extraction with cuDNN + attention
> MIL + self-supervised pretraining). We ship a faithful, self-contained
> **reduced-scope** version: the **attention-MIL classification head** — the exact
> algorithm CLAM uses for the slide-level call — running on the GPU over a bag of
> **pre-extracted** tile features. The CNN encoder is treated as an upstream black
> box (the sample *gives* us the features); THEORY.md §"Where this sits in the real
> world" describes the full pipeline.

## Summary

A whole-slide image (WSI) is a multi-gigapixel scan of a tissue slide. It is far
too big to classify pixel-by-pixel, so the standard approach (CLAM, ABMIL) chops
it into thousands of small **tiles**, turns each tile into a feature vector with a
frozen neural encoder, and then decides the slide's diagnosis from that **bag** of
features using **attention-based multiple-instance learning (MIL)**. This project
implements that attention-MIL head on the GPU: for each tile it computes an
**attention weight** (how diagnostically relevant the tile is), pools the tiles
into one slide embedding, and outputs a slide-level score. On a synthetic slide
with a few planted "tumor" tiles, the model's attention concentrates on exactly
those tiles and the slide is called `TUMOR` — a miniature of how real weakly-
supervised pathology models work.

## What this computes & why the GPU helps

Whole-slide images (WSIs) scanned at 40× magnification produce multi-gigapixel TIFF pyramids (0.5–5 GB per slide). Analysis requires GPU-accelerated tile extraction, feature extraction via pretrained CNNs (ResNet, ViT), and weakly supervised classification with attention-based multiple-instance learning (MIL). The tiling step alone for 10,000 slides produces ~500 million 224×224 patches; GPU DataLoaders must pipeline tile decompression, normalization, and augmentation to prevent GPU starvation. Spatial transcriptomics integration adds genomic annotations per spatial position, requiring co-registration of histology and sequencing data — a second-order GPU workload.

**The parallel bottleneck:** a slide is a **bag of N tiles** (tens of thousands per
slide, millions per cohort), and the attention head does the *same small
computation independently for every tile*: project the tile's feature vector
through a gated-attention network to a scalar logit. That per-tile projection is
embarrassingly parallel — **one GPU thread per tile**. The two cross-tile steps
(softmax over the logits, and the attention-weighted **pooling** of features into
the slide embedding) are **reductions**; the pooling is a scatter-reduction done
with `atomicAdd`, made deterministic by accumulating in **fixed-point integers**.

## The algorithm in brief

- **Gated-attention MIL** (Ilse et al. 2018; the CLAM/ABMIL head): per-tile
  attention logit `e_i = w·(tanh(V·h_i) ⊙ sigmoid(U·h_i))`.
- **Softmax** over the N tiles → attention weights `a_i ≥ 0`, `Σ a_i = 1`.
- **Attention pooling**: slide embedding `z = Σ_i a_i·h_i` (fixed-point atomic sum).
- **Linear classifier**: slide logit `s = w_c·z + b_c`, probability `sigmoid(s)`.

Upstream (not reimplemented here; see THEORY): tissue detection (Otsu), tiling,
and per-tile CNN/ViT feature extraction. See [THEORY.md](THEORY.md) for the full
science → math → algorithm → GPU-mapping derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/digital-pathology-whole-slide-image-analysis.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/digital-pathology-whole-slide-image-analysis.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\digital-pathology-whole-slide-image-analysis.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, prints the result, shows the
GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/` — a tiny, offline input so the demo runs
  with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` (documented, idempotent).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: TCGA (The Cancer Genome Atlas) slides — access via GDC Data Portal (https://portal.gdc.cancer.gov/); CAMELYON16/17 lymph node metastasis detection (https://camelyon17.grand-challenge.org/); PanCancer Atlas WSIs via TCGA; TUPAC16 tumor proliferation.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): the
pooled slide embedding, the slide logit and tumor probability, the top-attention
tile, a top-5 attention ranking, the `@0.5` slide call (`TUMOR` for the sample),
and `RESULT: PASS`. The program runs the attention-MIL forward pass on both the
**GPU** (`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`) and
asserts they agree within **1e-9** (see *Limitations* for why not bit-exact) —
that agreement is the correctness guarantee. The deterministic result goes to
**stdout** (diffed by the demo); timing and raw error magnitudes go to **stderr**.

## Code tour

Read in this order:

1. [`src/wsi.h`](src/wsi.h) — the shared `__host__ __device__` per-tile math
   (attention logit, sigmoid, and the fixed-point quantiser) that makes CPU and
   GPU agree. **Start here** — it is the heart of the project.
2. [`src/main.cu`](src/main.cu) — loads the slide bag, runs CPU + GPU, verifies, reports.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
4. [`src/kernels.cu`](src/kernels.cu) — the two kernels (logits, fixed-point pool) + host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the loader, the frozen model, the serial baseline.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

CLAM (https://github.com/mahmoodlab/CLAM) — GPU-accelerated attention MIL for WSI classification, standard baseline; OpenSlide Python (https://openslide.org/) — library for reading WSI file formats; HistomicsTK (https://github.com/DigitalSlideArchive/HistomicsTK) — GPU-accelerated WSI analysis toolkit; UNI pathology foundation model (https://github.com/mahmoodlab/UNI) — pretrained ViT on 100k WSIs.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Per-tile projection + softmax + atomic (fixed-point) reduction.** One thread per
tile computes the tile's attention logit (independent, constant-memory model
weights broadcast to every thread). A small deterministic softmax runs on the host
(exact, reused by both paths). A second kernel does the attention-weighted pooling
as a `atomicAdd` scatter-reduction in **fixed-point integers**, so the sum is
order-independent and matches the CPU exactly. This is the same
independent-jobs + atomic-reduce shape as flagships `11.09` (k-means) and `1.12`
(Tanimoto). The catalog's full-pipeline pattern (cuDNN feature extraction, DALI
decode, cuBLAS attention, multi-GPU) is described in THEORY; here the head is the
self-contained GPU kernel.

## Exercises

1. **Softmax temperature.** Divide the logits by a temperature `T` before the
   softmax. How does `T→0` (sharp) vs `T→∞` (uniform) change the attention map and
   the slide probability? (Relates directly to `w[0]` in `default_params()`.)
2. **Bigger bags.** Run `python scripts/make_synthetic.py --n 20000` and re-run.
   Watch the GPU/CPU timing gap on stderr grow as the tile count rises.
3. **Sweep the tumor fraction.** Generate slides at `--tumor-frac 0, 0.02, 0.05,
   0.1, 0.3` and plot the output probability. You have just drawn an ROC-style
   sensitivity curve for the (frozen) model.
4. **A GPU softmax.** Move the softmax reduction (currently host-side) into a
   kernel using a block-wide `__shfl_down_sync` reduction for the max and sum.
   Keep it deterministic — why is a *tree* reduction over doubles reproducible
   while a naive `atomicAdd` of floats is not?
5. **Second attention head.** Add a second hidden unit that detects a *different*
   feature pattern and combine the two logits. This is the multi-class CLAM idea.

## Limitations & honesty

- **Reduced scope.** This is the attention-MIL *head* only. The CNN/ViT feature
  extractor, tissue detection (Otsu), stain normalization (Macenko/Vahadane), and
  self-supervised pretraining (DINO/MAE) named in the catalog are **not**
  implemented — THEORY.md describes them and where they fit.
- **Frozen, hand-set weights.** The attention model is not *trained*; its weights
  are hand-chosen so the demo is reproducible and interpretable. A real model
  learns them from labeled slides by backpropagation.
- **Synthetic data.** The tile features are synthetic Gaussian "tumor"/"background"
  vectors (labeled synthetic everywhere), not real histology. Nothing here is
  diagnostic. **Not for clinical use.**
- **Not bit-exact.** GPU `tanh`/`exp` differ from host libm by ~1 ULP, so attention
  weights differ by ~1e-16; the fixed-point pooling absorbs that so the embedding
  and probability match to ~1e-9. We verify to 1e-9 and say so — we do not pretend
  the two are bit-identical.
