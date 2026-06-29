# 4.11 — Digital Pathology / Whole-Slide Image Analysis

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.11`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

<!-- =======================================================================
     SCAFFOLD STATUS: this README was stamped from the catalog. The prose
     fields below (Deep dive / Algorithms / Datasets / Prior art) are filled
     in from the catalog. Sections marked TODO(impl)/TODO(theory) must be
     completed by the project author before this project is "done"
     (see CLAUDE.md §4.1 and tools/verify_project.py).
     ======================================================================= -->

## Summary

TODO(impl): One paragraph, plain language — what this project does and why a
learner should care. (Seed from the deep dive below.)

## What this computes & why the GPU helps

Whole-slide images (WSIs) scanned at 40× magnification produce multi-gigapixel TIFF pyramids (0.5–5 GB per slide). Analysis requires GPU-accelerated tile extraction, feature extraction via pretrained CNNs (ResNet, ViT), and weakly supervised classification with attention-based multiple-instance learning (MIL). The tiling step alone for 10,000 slides produces ~500 million 224×224 patches; GPU DataLoaders must pipeline tile decompression, normalization, and augmentation to prevent GPU starvation. Spatial transcriptomics integration adds genomic annotations per spatial position, requiring co-registration of histology and sequencing data — a second-order GPU workload.

**The parallel bottleneck:** TODO(impl) — name the specific step that is
parallelized on the GPU and why it dominates the runtime.

## The algorithm in brief

Attention-based MIL (CLAM, ABMIL), patch-level feature extraction (ResNet-50, ViT, UNI foundation model), stain normalization (Macenko, Vahadane), Otsu thresholding for tissue detection, tumor microenvironment clustering (DINO, MAE pretraining), survival prediction.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

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

Success looks like `demo/expected_output.txt`. The program computes the result on
both the **GPU** (`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`)
and asserts they agree within the documented tolerance — that agreement is the
correctness guarantee.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads data, runs CPU + GPU, verifies, reports.
2. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread-mapping idea.
3. [`src/kernels.cu`](src/kernels.cu) — the kernel(s) and host wrapper.
4. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline.
5. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

CLAM (https://github.com/mahmoodlab/CLAM) — GPU-accelerated attention MIL for WSI classification, standard baseline; OpenSlide Python (https://openslide.org/) — library for reading WSI file formats; HistomicsTK (https://github.com/DigitalSlideArchive/HistomicsTK) — GPU-accelerated WSI analysis toolkit; UNI pathology foundation model (https://github.com/mahmoodlab/UNI) — pretrained ViT on 100k WSIs.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

cuDNN (ResNet/ViT feature extraction per tile); DALI (GPU tile decode/augment pipeline); GPU-resident attention matrix for MIL (cuBLAS); batched tile inference with pinned memory transfer; multi-GPU feature extraction with `torch.multiprocessing`. --

## Exercises

TODO(impl): 3–5 "try this next" extensions for the learner. Ideas to seed from:
larger inputs, a second precision (FP64), shared-memory tiling, a different
block size sweep, or an additional verification metric.

## Limitations & honesty

TODO(impl): What is simplified, what is synthetic, what would differ in
production. Be explicit — this is study material, not a clinical tool.
