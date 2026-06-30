# 4.7 — Medical Image Segmentation (Deep Learning)

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟢 Beginner · Established** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.7`
>
> _Educational only — not for clinical use (see CLAUDE.md §8). This ships a
> **reduced-scope teaching version**: a fixed-weight 3D-convolution segmentation
> head, not a trained nnU-Net. See [THEORY.md](THEORY.md) §7 for the difference._

## Summary

Volumetric **segmentation** labels every voxel of a 3D CT/MRI scan as part of an
anatomical structure (organ, tumor, vessel) or background. Production tools do
this with deep 3D convolutional networks (3D U-Net, nnU-Net, MONAI, Swin-UNETR);
those networks spend the overwhelming majority of their compute in one primitive:
the **3D convolution**. This project isolates that primitive. It runs a tiny,
**fully deterministic** 2-layer fully-convolutional head — a Gaussian *denoise*
followed by a local-mean *threshold*, both expressed as 3×3×3 conv layers — that
segments a bright "lesion" from a synthetic volume, on the GPU (one thread per
output voxel) and on a CPU reference, and verifies they agree to the bit. Because
the synthetic lesion's location is known, it reports a real **Dice accuracy**
(≈ 0.96), so you see the network actually find the structure — not just that two
implementations match.

## What this computes & why the GPU helps

A 3D convolution turns an input volume into an output volume; **each output voxel
is an independent dot product** of a small weight stencil (here 3×3×3 = 27 taps)
with a local neighbourhood, summed over input channels. Our head stacks two such
layers (1 → 2 → 2 channels) with a ReLU between them and a per-voxel argmax at the
end to produce the 0/1 mask.

**The parallel bottleneck:** the convolution itself. There are millions of output
voxels and they are mutually independent, so we give **one GPU thread per output
voxel** — the canonical 3D-stencil/gather workload that cuDNN accelerates inside
real U-Nets. The catalog notes a `512×512×200` CT through a 3D U-Net is ~200
GFLOPs per forward pass: ~40–50 min on a CPU versus 20–50 s on a GPU
(TotalSegmentator, 117 structures). Our volume is tiny (so it fits in a slide and
is honestly *launch-bound*), but the parallel structure is identical.

## The algorithm in brief

- **Layer 1** — 3×3×3 conv (1 input channel → 2 hidden), then ReLU. Channel 0 is a
  Gaussian smoother (denoise); channel 1 is identity (kept for experiments).
- **Layer 2** — 3×3×3 conv (2 channels → 2 classes). Class 1 (lesion) is a uniform
  box-average over the smoothed channel minus a threshold; class 0 (background) is
  a constant 0.
- **Argmax** over the 2 class logits → integer label map `{0,1}`.
- **Dice** of the predicted mask vs. the known ground-truth sphere = accuracy.

Full key-algorithms list (catalog): 3D U-Net, nnU-Net, Swin-UNETR, TransUNet,
DeepMedic, V-Net, residual encoder-decoder, cascaded networks, multi-scale feature
pyramid, CRF post-processing, semi-supervised pseudo-labeling. This teaching
version implements the **3D-conv forward pass** that underlies all of them; see
[THEORY.md](THEORY.md) §7 for what the full pipeline adds.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/medical-image-segmentation-deep-learning.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/medical-image-segmentation-deep-learning.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\medical-image-segmentation-deep-learning.sln /p:Configuration=Release /p:Platform=x64
```

This project links only the CUDA runtime (`cudart`); no extra CUDA libraries are
needed (the convolution is hand-rolled on purpose — no black box).

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (CMake build)
```

The demo builds if needed, runs on the committed synthetic volume in
`data/sample/`, prints the predicted lesion-voxel count, the **Dice** vs. ground
truth, a central-slice ASCII mask, and the GPU-vs-CPU agreement check; timing goes
to stderr. It then diffs stdout against `demo/expected_output.txt`.

## Data

- **Sample (committed):** `data/sample/volume_sample.txt` — a tiny **synthetic**
  12×16×16 CT-like volume (a noisy bright lesion sphere on soft-tissue background)
  plus its ground-truth mask, so the demo runs offline with zero downloads.
- **Generate / resize:** `python scripts/make_synthetic.py [--D .. --H .. --W ..]`.
- **Full datasets:** `scripts/download_data.ps1` / `.sh` print pointers (most real
  sets require registration and are not redistributable — the sample is synthetic).
- **Provenance, layout & license:** see [data/README.md](data/README.md).

Catalog dataset notes: Medical Segmentation Decathlon (http://medicaldecathlon.com/);
TotalSegmentator training set (https://zenodo.org/record/6802614); KiTS23
(https://kits-challenge.org/kits23/); BraTS (https://www.synapse.org/#!Synapse:syn27046444).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt):
133 predicted lesion voxels vs. 123 ground-truth, **Dice = 0.9609**, a circular
lesion cross-section in the central-slice ASCII mask, and
`RESULT: PASS`. The program computes the segmentation on both the **GPU**
(`src/kernels.cu`) and a **CPU reference** (`src/reference_cpu.cpp`) and asserts
the integer label maps are **identical** and the float logits agree within
`1e-3`. That agreement, plus the Dice against known ground truth, is the
correctness guarantee. stdout is deterministic (integer counts only); timing is on
stderr.

## Code tour

Read in this order:

1. [`src/reference_cpu.h`](src/reference_cpu.h) — the `Volume`/`SegNet` types and
   **the shared `__host__ __device__` core** (`conv3x3x3_at`, `relu`) that both
   CPU and GPU call, so their math is identical.
2. [`src/main.cu`](src/main.cu) — loads the volume + ground truth, runs CPU + GPU,
   verifies, scores Dice, reports.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-
   voxel mapping idea.
4. [`src/kernels.cu`](src/kernels.cu) — the two voxel-parallel conv kernels,
   constant-memory weights, and host wrapper.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the loader, the fixed-weight
   `make_segnet`, the serial forward pass, and Dice.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **nnU-Net** (https://github.com/MIC-DKFZ/nnUNet) — self-configuring U-Net,
  2D/3D, GPU train + inference. Learn how it auto-picks patch/batch size and
  topology from a dataset fingerprint.
- **TotalSegmentator** (https://github.com/wasserth/TotalSegmentator) — 117-class
  whole-body CT, GPU inference in < 1 min. The production target this gestures at.
- **MONAI** (https://github.com/Project-MONAI/MONAI) — PyTorch medical-AI
  framework; study its GPU-resident transforms and network zoo.
- **Swin-UNETR** (https://github.com/Project-MONAI/research-contributions) —
  transformer-based 3D segmentation; the attention alternative to pure conv.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**3D stencil / gather, one thread per output voxel**, with the (tiny, read-only)
filter weights in **constant memory** (broadcast to every thread). This is the
hand-rolled cousin of the catalog's stated pattern — cuDNN 3D convolutions, Tensor
Cores (FP16/BF16), Unified Memory for large volumes, sliding-window patch
inference, DDP+NCCL multi-GPU, GPU-resident augmentation (MONAI/DALI). We hand-roll
the 27-tap dot product so it is not a black box; [THEORY.md](THEORY.md) §4 explains
what cuDNN does instead (im2col/Winograd → GEMM on Tensor Cores).

## Exercises

1. **Shared-memory tiling.** The current kernel re-reads each input voxel ~27×
   from global memory. Tile a block of the volume + a halo into `__shared__`
   memory (as flagship `7.10` does in 1-D) and read the neighbourhood from there.
   Measure the speedup at a larger volume (`--D 64 --H 64 --W 64`).
2. **Tune the threshold.** `LESION_THRESHOLD` (in `reference_cpu.cpp`) trades
   precision vs. recall. Sweep it and plot Dice; find the value that maximizes
   Dice on `make_synthetic.py --noise 0.05`. Why does more noise shift the optimum?
3. **A real edge/blob filter.** Replace the box-average in class 1 with a
   Laplacian-of-Gaussian (center-surround) filter so the head responds to *shape*,
   not just brightness, and segment a lesion that is the *same* intensity as
   tissue but more compact.
4. **Add a third class.** Extend `N_CLASS` to 3 (background / lesion / bright-edge)
   and design the layer-2 filters + biases. Watch the argmax tie-break still keep
   the result deterministic.
5. **Block-size sweep.** Try `SEG_BLOCK ∈ {64, 128, 256, 512}` on a large volume
   and record kernel time. Explain the occupancy trade-off you observe.

## Limitations & honesty

- **Reduced scope, by design.** The weights are **hand-set, not learned**; there
  is no training, no encoder/decoder, no skip connections, just two conv layers.
  It segments one bright structure via denoise-then-threshold. A trained nnU-Net
  differs in depth, channels, receptive field, and class count — but uses the
  *same* 3D-conv primitive (see [THEORY.md](THEORY.md) §7).
- **Synthetic data.** The volume and its labels are generated math, labelled
  synthetic everywhere. The Dice ≈ 0.96 reflects how cleanly the synthetic lesion
  is separated by intensity; it is **not** a clinical accuracy claim.
- **No real-format I/O.** Real volumes are NIfTI/DICOM; we use a plain text volume
  so the loader stays readable. Wiring a NIfTI reader is left out of scope.
- **Tiny, launch-bound timing.** On a 3072-voxel toy the GPU time is dominated by
  launch overhead — a *teaching artifact*, not a benchmark. The GPU's edge appears
  at real `512³` volumes with hundreds of channels.
- **Not for clinical use.** No output here is a diagnosis or a treatment decision.
