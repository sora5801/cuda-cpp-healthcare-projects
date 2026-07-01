# 4.24 — CT/MRI Super-Resolution

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.24`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

This project super-resolves a grayscale medical-style image **2× in each axis**
using the exact upsampling operator at the heart of ESRGAN/ESPCN: a small learned
**feature convolution + ReLU**, then a **sub-pixel convolution** whose `R²` output
channels are rearranged by **pixel-shuffle** into the high-resolution image. Every
high-res output pixel is computed **independently** by one GPU thread that gathers
a tiny 3×3 low-res neighbourhood — the classic imaging *gather* pattern. This is a
**reduced-scope teaching version** (CLAUDE.md §13): the two-layer network runs with
**fixed, synthetic weights** (a bilinear-interpolation + edge-sharpening design)
rather than trained ones, so the demo is fully deterministic. The compute path is
identical to a deployed SR inference kernel, and the network still beats naive
nearest-neighbour upscaling by **+1.23 dB PSNR** on the sample.

## What this computes & why the GPU helps

Clinical CT/MRI is acquired at sub-optimal resolution due to dose constraints, scan time, or scanner capability; super-resolution (SR) enhances images 2–4× isotropically using deep neural networks. For MRI, anisotropic SR (thick slice → isotropic) upsamples a 5 mm axial slice to 1 mm coronal/sagittal using networks trained on pairs of high/low-resolution volumes. GANs (ESRGAN-Med, CycleGAN) generate perceptually sharp images; diffusion SR models produce hallucination-free probabilistic outputs. Processing a 512×512×100 CT volume through a 3D ESRGAN requires ~500 GFLOPS per forward pass; clinical deployment requires TensorRT-optimized inference at <5 s/volume on a single GPU.

**The parallel bottleneck:** the SR forward pass evaluates, for **every** high-res
output pixel, a small convolution over its low-res neighbourhood. A 512×512×100
volume upscaled 2× is ~200 million output voxels, each doing hundreds of
multiply-adds — hundreds of GFLOPs of *fully independent* work. That maps perfectly
to the GPU: **one thread per output pixel/voxel**, no synchronization, memory-bound
reads served by the read-only/constant caches. Our teaching kernel implements
exactly that mapping on a 2-D image.

## The algorithm in brief

Full-fidelity SR spans many methods (from the catalog): ESRGAN (enhanced SRGAN), 3D U-Net SR, CycleGAN for unpaired SR, diffusion model SR (SR3, DDPM), subpixel convolution (ICNR), attention U-Net SR, learned upsampling (LIIF), perceptual and adversarial losses, self-supervised SR.

This project implements the **sub-pixel-convolution upsampler** (Shi et al. 2016),
the building block ESRGAN stacks to reach 4×:

- **Degradation model:** LR = 2×2 box-average downsample of a ground-truth HR image.
- **Layer 1 — feature conv + ReLU:** `C_FEAT` learned 3×3 filters over the LR image.
- **Layer 2 — sub-pixel conv:** a 3×3 conv mapping features → `R²` channels.
- **Pixel-shuffle (depth-to-space):** scatter the `R²` channels into an `R×R` HR
  block. We do this *implicitly by indexing* (pixel `(hx,hy)` selects channel
  `(hy%R)*R + (hx%R)`), which is the same math without a physical reshuffle.
- **Verification:** GPU output vs. CPU reference (exact), plus **PSNR** vs. ground
  truth and vs. nearest-neighbour (the science check).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/ct-mri-super-resolution.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/ct-mri-super-resolution.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\ct-mri-super-resolution.sln /p:Configuration=Release /p:Platform=x64
```

Both `Release|x64` and `Debug|x64` build with **zero warnings**. Only the CUDA
runtime (`cudart_static.lib`) is linked — no extra CUDA libraries are needed for
this hand-rolled convolution.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (CMake build)
```

The demo builds if needed, runs on `data/sample/phantom_hr.txt`, prints the result,
shows the GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/phantom_hr.txt` — a tiny **synthetic** 32×32
  ground-truth HR "phantom" (smooth blobs + resolution bars + a sharp ring). The
  program degrades it 2× to make the LR input, then super-resolves it back. Runs
  offline with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print links + instructions
  for the real datasets (they require registration; the scripts never bypass it).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: HCP 7T/3T paired brain MRI (https://db.humanconnectome.org/); fastMRI (https://fastmri.med.nyu.edu/) — implicitly used for SR evaluation; IXI brain MRI dataset (https://brain-development.org/ixi-dataset/); MSD CT tasks for resolution enhancement.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt):

```
4.24 -- CT/MRI Super-Resolution
scale R=2  |  LR 16x16 -> HR 32x32  |  net: 4 feat ch, 3x3 conv + subpixel
PSNR nearest-neighbour vs truth = 22.5583 dB
PSNR super-resolved   vs truth = 23.7905 dB
PSNR improvement over baseline = 1.2322 dB
HR samples (8 evenly spaced): 0.198325 0.821572 0.512523 0.610395 0.660512 0.370428 0.288678 0.155375
RESULT: PASS (GPU matches CPU within tol=1.0e-06)
```

The program computes the result on both the **GPU** (`src/kernels.cu`) and a **CPU
reference** (`src/reference_cpu.cpp`) and asserts they agree within `1e-6` — that
agreement is the correctness guarantee. They call the *same* `sr_hr_pixel()` in
`src/sr_core.h`, so the match is essentially bit-exact. The positive **PSNR
improvement over baseline** confirms the network reconstructs a sharper image than
block-replication. (stdout is deterministic; timing goes to stderr.)

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the image, degrades it, runs CPU + GPU SR, verifies, reports.
2. [`src/sr_core.h`](src/sr_core.h) — **the heart**: the `__host__ __device__` per-pixel network math (feature conv, ReLU, sub-pixel conv, pixel-shuffle indexing). Shared verbatim by CPU and GPU.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-HR-pixel mapping.
4. [`src/kernels.cu`](src/kernels.cu) — the kernel (calls `sr_hr_pixel`) and the upload/launch/download wrapper; weights in constant memory.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — loader, the **synthetic weight design**, degradation, serial reference, PSNR.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

MONAI SR examples (https://github.com/Project-MONAI/MONAI) — 3D medical SR reference implementations; BasicSR (https://github.com/XPixelGroup/BasicSR) — general GPU SR framework adaptable to medical; SynthSR (https://github.com/BBillot/SynthSR) — multi-contrast MRI SR/synthesis; MedSRGAN (search GitHub for "medical image super resolution GAN").

- **ESPCN** (Shi et al. 2016, *Real-Time SISR Using an Efficient Sub-Pixel CNN*) — the origin of the sub-pixel/pixel-shuffle upsampler implemented here.
- **ESRGAN** (Wang et al. 2018) — stacks these sub-pixel blocks with a GAN loss for perceptual sharpness.
- **cuDNN / TensorRT** — how production runs the real (wide, trained) convolutions and pixel-shuffle on Tensor Cores; see "CUDA pattern" below.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Imaging gather: one thread per output pixel** (PATTERNS.md §1, exemplar 4.01 CT
backprojection). Each thread maps its HR coordinate to an LR cell + sub-pixel phase,
gathers a 3×3 LR window, and evaluates the tiny network — no shared memory, no
atomics, no synchronization. Network **weights live in `__constant__` memory**
(broadcast cache: every thread reads the same weights). The per-pixel math is a
shared `__host__ __device__` function (PATTERNS.md §2) so CPU and GPU agree exactly.

In production the catalog lists: cuDNN (3D transposed convolutions, pixel shuffle); Tensor Cores for FP16 SR training; gradient penalty in discriminator (cuBLAS); CuPy for efficient patch extraction; TensorRT for INT8/FP16 inference deployment. This teaching version hand-rolls the convolution instead of calling cuDNN, so the arithmetic is fully visible (no black box).

## Exercises

1. **Go to 4×.** Change `SR_SCALE` to 4 in `sr_core.h`, regenerate the weights
   (the bilinear offset fraction `f` becomes `1/8` and there are 16 sub-pixel
   phases), rebuild, and compare PSNR. What happens to the gap over nearest-neighbour?
2. **Materialize the feature map.** The kernel recomputes layer-1 features on the
   fly (per HR pixel). Add a first kernel that writes the `C_FEAT`-channel LR
   feature map to global memory, then a second that reads it — measure the speedup
   and reason about arithmetic-vs-memory trade-offs (THEORY §GPU-mapping).
3. **Shared-memory tiling.** Stage each block's LR tile (+1-pixel halo) into shared
   memory once, like the 1-D conv in flagship 7.10. Does it help at this size? When?
4. **A real slice.** Extract one axial slice from an IXI MRI volume (link in
   `data/README.md`), normalize to `[0,1]`, and run SR on it. Report PSNR/SSIM.
5. **Learn the weights.** Replace `make_sr_weights()` with weights fit by least
   squares on (LR, HR) patch pairs from the phantom — a one-layer "trained" SR — and
   see whether it beats the hand-designed bilinear+unsharp design.

## Limitations & honesty

- **Reduced scope.** This is the *sub-pixel-conv upsampler*, not a full ESRGAN /
  diffusion SR model. There is no GAN, no perceptual loss, no 3-D volume, no
  training loop. THEORY.md "Where this sits in the real world" describes the gap.
- **Synthetic, fixed weights.** The network is **not trained**; its weights are a
  hand-designed bilinear-interpolation + Laplacian-unsharp operator. It is labeled
  synthetic everywhere. A trained model would generally do better and could also
  *hallucinate* detail — a real clinical risk this toy avoids by construction.
- **Synthetic data.** The sample is a generated phantom, not a real scan. PSNR here
  measures recovery of a known degradation, not diagnostic quality.
- **Not for clinical use.** Nothing here may inform diagnosis or treatment
  (CLAUDE.md §8). Super-resolution can invent plausible-but-false structure;
  clinical SR requires rigorous validation this project does not attempt.
