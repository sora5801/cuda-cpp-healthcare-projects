# 4.9 — Image Denoising & Restoration

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.9`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Medical scans are noisy: low-dose CT is grainy with quantum (Poisson) noise, MRI
carries thermal noise, ultrasound has speckle. **Denoising** recovers a cleaner
image without smearing the edges that carry the diagnosis. This project implements
**Non-Local Means (NLM)** — the classic patch-based denoiser — as a custom CUDA
kernel. NLM replaces each pixel with a weighted average of pixels whose *local
neighbourhood (patch)* looks similar, so it averages away noise while preserving
edges and texture. It is embarrassingly parallel: every output pixel is
independent, so we give each one its own GPU thread. On a synthetic phantom the
demo raises image quality by **~7.9 dB PSNR**, and the GPU result matches a plain
CPU reference to `2.4e-7`.

## What this computes & why the GPU helps

Medical images suffer from quantum noise (CT, PET, X-ray), thermal noise (MRI), and speckle (ultrasound). Deep denoising networks (DnCNN, RED-CNN for low-dose CT, Noise2Void for unsupervised fluorescence) process 2D or 3D patches through many conv layers, requiring substantial FLOPS and large GPU memory for 3D volumetric batches. Diffusion-model denoisers now achieve state-of-the-art perceptual quality but require iterative reverse-diffusion steps (50–1,000 denoising steps), each a full forward pass through a large UNet, making GPU mandatory. Non-learning methods (NLM, BM4D) have O(N²) complexity in voxel count, acceleratable via CUDA block-matching and nearest-neighbor search.

**The parallel bottleneck (what this project accelerates):** the NLM *block
matching*. For every output pixel we scan a search window of candidate pixels and,
for each candidate, compare two small patches by summing their squared per-pixel
differences. The cost is `O(P · (2S+1)² · (2R+1)²)` for `P` pixels, search radius
`S`, patch radius `R` — billions of multiply-adds for even a modest clinical
slice. But **each output pixel only reads the noisy input and writes its own
result**, so there are no dependencies between pixels: we map the 2-D image onto a
2-D thread grid, one thread per output pixel, no locks and no atomics. This is
exactly the catalog's named pattern — *"custom CUDA for NLM block matching (each
thread computes patch distance vs. all neighbors)."*

## The algorithm in brief

- **Non-Local Means (Buades–Coll–Morel, 2005):** `out(p) = Σ_q w(p,q)·in(q) / Σ_q w(p,q)`.
- **Patch distance** `d²(p,q)` = mean squared difference of the two patches centred at `p` and `q`.
- **Weight** `w(p,q) = exp(−max(d²−2σ², 0)/h²)` — similar patches get weight ≈ 1, dissimilar ≈ 0.
  - `σ` is the noise level; subtracting `2σ²` is the standard noise-bias correction.
  - `h` is the filter strength (larger `h` → more smoothing).
- **Border handling:** mirror (reflect) so patches near the edge stay in-bounds, identically on CPU and GPU.
- **Quality metric:** PSNR in dB vs. the (synthetic) clean image — a working denoiser *raises* PSNR.

The full family in the catalog also includes DnCNN, RED-CNN, Noise2Void/Noise2Self,
score-based diffusion (DDPM/DDIM), BM3D/BM4D, wavelet shrinkage, and total-variation
denoising. We implement classic NLM because it is the one that (a) needs *no* trained
weights or external library, (b) is a clean, self-contained CUDA teaching kernel, and
(c) directly matches the catalog's "custom CUDA block matching" pattern. See
[THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation, and how the deep-learning and diffusion methods differ.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/image-denoising-restoration.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/image-denoising-restoration.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\image-denoising-restoration.sln /p:Configuration=Release /p:Platform=x64
```

Both `Debug|x64` and `Release|x64` build with zero warnings. No extra CUDA library
is linked — NLM is a hand-written kernel, so only the CUDA runtime
(`cudart_static.lib`) is needed.

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/phantom_sample.txt`, prints the
result, shows the GPU-vs-CPU agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/phantom_sample.txt` — a tiny 32×32 **synthetic**
  phantom (bright disk + square inset on a dark field) with additive Gaussian noise,
  plus the clean ground truth for PSNR scoring. Runs the demo offline, zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print instructions/links for
  the real medical datasets (they do **not** bypass any registration).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: 2016 AAPM Low-Dose CT Challenge (https://www.aapm.org/grandchallenge/lowdosect/) — quarter-dose / full-dose pairs; NLST (National Lung Screening Trial) via TCIA; Fluorescence Microscopy Noise Dataset (https://github.com/juglab/n2v) — for Noise2Void; SIDD smartphone noise dataset (image domain).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt):

```
4.9 -- Image Denoising & Restoration (Non-Local Means)
Non-Local Means denoise: 32x32 image, patch r=2, search r=5, sigma=0.080, h=0.096
PSNR noisy  vs clean = 22.0857 dB
PSNR denoised vs clean = 29.9900 dB
PSNR improvement = 7.9043 dB
denoised central row (8 samples): 0.1317 0.1689 0.7330 0.7455 0.7630 0.7643 0.6591 0.1688
RESULT: PASS (GPU matches CPU within tol=1.0e-04)
```

The program denoises on both the **GPU** (`src/kernels.cu`) and a **CPU reference**
(`src/reference_cpu.cpp`) and asserts they agree within `1.0e-4` — that agreement is
the correctness guarantee. Both call the *same* per-pixel math in
[`src/nlm_core.h`](src/nlm_core.h), so the observed error is only float FMA rounding
(`~2e-7`). The PSNR jump (22.1 → 30.0 dB) shows the denoiser actually works; the
central-row profile shows the dark field (~0.13–0.17), the bright disk (~0.73–0.76),
and the darker square inset (the `0.6591` dip) all cleanly recovered.

## Code tour

Read in this order:

1. [`src/nlm_core.h`](src/nlm_core.h) — **start here**: the shared `__host__ __device__`
   per-pixel NLM math (patch distance, weight, weighted mean). One copy, used by both sides.
2. [`src/main.cu`](src/main.cu) — loads the image, runs CPU + GPU, verifies, reports PSNR.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the one-thread-per-pixel idea.
4. [`src/kernels.cu`](src/kernels.cu) — the `nlm_kernel` and its host wrapper (malloc/copy/launch/time).
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline + loader + PSNR.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, host I/O helpers.

## Prior art & further reading

N2V / Noise2Void (https://github.com/juglab/n2v) — self-supervised GPU denoising for microscopy and MRI; MONAI model zoo — RED-CNN and DnCNN for CT; DnCNN PyTorch (https://github.com/cszn/DnCNN) — GPU-accelerated Gaussian denoiser; DiffusionMBIR (https://github.com/HJ-harry/DiffusionMBIR) — score-based diffusion for CT reconstruction/denoising.

One line each on what to learn:
- **Buades, Coll & Morel (2005), "A non-local algorithm for image denoising"** — the paper this kernel implements; read it for the weight derivation.
- **DnCNN (cszn/DnCNN)** — how a learned CNN residual-predicts the noise; the modern successor to NLM for Gaussian denoising.
- **RED-CNN (MONAI)** — an encoder–decoder tuned for low-dose CT; shows the clinical framing.
- **Noise2Void (juglab/n2v)** — self-supervised denoising when you have no clean targets (common in microscopy/MRI).
- **DiffusionMBIR** — score-based diffusion as a denoising *prior* inside CT reconstruction; the current SOTA-quality direction.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Per-output-pixel gather + block matching** (no external CUDA library). One GPU
thread owns one output pixel; a 2-D `16×16` thread grid tiles the image. Each thread
loops over its search window, computes a patch distance against every candidate,
exponentiates it into a weight, and accumulates `Σ w·value` and `Σ w` in registers —
no shared memory and no atomics, because output pixels are independent. This mirrors
the CT-backprojection flagship (`4.01`), which is the same "per-pixel gather" idiom.
The catalog also lists cuDNN (for DnCNN/RED-CNN convolutions), cuBLAS (dense layers),
and TensorRT/FP16 (clinical deployment) for the *learned* denoisers — out of scope
here because they require trained weights; THEORY.md explains where they fit.

## Exercises

1. **Turn the noise up.** Regenerate the sample with `python scripts/make_synthetic.py --sigma 0.15`
   (and re-capture `expected_output.txt`). How much PSNR does NLM recover now? Does a larger `h` help?
2. **Shared-memory tiling.** Neighbouring threads re-read overlapping input. Stage each
   block's input tile *plus its halo* into `__shared__` memory and read patches from there.
   Measure the speed-up (THEORY.md "GPU mapping" sketches this).
3. **Bigger search window.** Bump `--search-radius` to 8. NLM cost grows as `(2S+1)²` — watch the
   CPU time explode while the GPU stays cheap. Plot CPU-vs-GPU time as `S` grows.
4. **A second metric.** Add SSIM (structural similarity) alongside PSNR; it correlates better
   with perceived quality on edges. Print it on stdout and keep it deterministic.
5. **Precompute the patch sums.** The naive kernel recomputes overlapping patch differences.
   Implement the integral-image / summed-area-table trick so each patch distance is `O(1)`.

## Limitations & honesty

- **Synthetic data.** The committed phantom and its Gaussian noise are **synthetic**
  (fixed-seed), labelled synthetic everywhere. Real CT noise is Poisson and correlated
  after reconstruction; real MRI noise is Rician. This is a *teaching* denoiser, **not a
  clinical tool** — no output here may inform any diagnosis or treatment.
- **Reduced scope.** We implement *classic* NLM, not the catalog's full method list. The
  learned denoisers (DnCNN, RED-CNN, Noise2Void) and diffusion models (DDPM/DDIM) need
  trained weights and cuDNN/TensorRT; they are described in THEORY.md but not built here
  (CLAUDE.md §13 — ship the tractable teaching version, document the rest).
- **Naive kernel.** No shared-memory tiling, no early-abort on the patch distance, no
  integral-image acceleration — all deliberately left as exercises so the core idea stays
  readable. Production NLM (and BM3D/BM4D) add all of these.
- **PSNR needs a clean reference** we only have because the data is synthetic; in the clinic
  there is no ground truth, so quality is judged by radiologists and task-based metrics.
