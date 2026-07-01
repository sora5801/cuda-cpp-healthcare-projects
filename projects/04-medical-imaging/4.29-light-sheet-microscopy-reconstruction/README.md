# 4.29 — Light-Sheet Microscopy Reconstruction

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.29`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

Light-sheet microscopy records 3D image stacks that are **blurred** by the
microscope's optics and **noisy** from faint photon counts. This project
**deblurs** one plane with **Richardson-Lucy (RL) deconvolution** — the classic,
Poisson-noise-optimal method — and does the heavy lifting (the convolutions)
in the **Fourier domain with NVIDIA's cuFFT** on the GPU. It is a **reduced-scope
teaching version** of the full multi-view, terabyte-scale LSFM reconstruction
pipeline: a single 2D plane, one view, an isotropic Gaussian point-spread function
(PSF). You get the exact inner loop that BigStitcher and DeconvolutionLab2 run —
re-blur, correction ratio, back-project, multiplicative update, all in k-space —
small enough to read end to end, verified against a plain CPU reference.

## What this computes & why the GPU helps

Light-sheet fluorescence microscopy (LSFM / SPIM) acquires terabyte-scale datasets
of developing embryos or cleared organs by illuminating a thin optical plane; the
resulting stacks must be deblurred (deconvolved), fused across views, and stitched
from tiles. **Multi-view Richardson-Lucy deconvolution** with a Gaussian PSF on a
10³ × 10³ × 10³ sub-volume needs **~10¹² multiply-accumulates per outer iteration**
— GPU-essential. This project isolates the deconvolution core on a 2D plane.

**The parallel bottleneck:** each RL iteration performs **two convolutions** of the
whole image. Done directly that is `O(N²)` per pixel; done via the **FFT** it is a
single element-wise multiply in frequency space (`O(N log N)` total). cuFFT runs
those FFTs on the GPU, and a handful of one-thread-per-pixel kernels do the
element-wise RL arithmetic around them. Convolution *is* the runtime, and the FFT
is what parallelizes it — that is the whole lesson.

## The algorithm in brief

- **Richardson-Lucy deconvolution** (Poisson maximum-likelihood, multiplicative,
  positivity-preserving, flux-conserving).
- **Fourier-domain convolution** via the convolution theorem:
  `h * x = IFFT(FFT(h)·FFT(x))`; the adjoint (back-projection) uses `conj(FFT(h))`.
- **cuFFT** double-precision `D2Z` / `Z2D` real↔complex transforms.
- A **shared `__host__ __device__` core** (`src/rl_core.h`) so the CPU reference and
  the GPU kernels run identical per-pixel math.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)). The project links
**cuFFT** (`cufft.lib`) — already wired into `build/*.vcxproj` and `CMakeLists.txt`.

1. Open `build/light-sheet-microscopy-reconstruction.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/light-sheet-microscopy-reconstruction.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\light-sheet-microscopy-reconstruction.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (CMake build)
```

The demo builds if needed, runs on `data/sample/lsfm_sample.txt`, prints the
reconstruction statistics, shows the GPU-vs-CPU agreement check, and prints a
timing line (GPU cuFFT vs CPU direct-DFT).

## Data

- **Sample (committed):** `data/sample/lsfm_sample.txt` — a **synthetic** 32×32
  plane (bright "beads" blurred by a Gaussian + mild deterministic noise) so the
  demo runs **offline with zero downloads**. Regenerate with
  `python scripts/make_synthetic.py`.
- **Full datasets:** `scripts/download_data.ps1` / `.sh` print the public LSFM
  sources (they do **not** fetch TBs; real LSFM data is 3D TIFF/HDF5/N5/Zarr and
  some needs registration).
- **Provenance & license:** see [data/README.md](data/README.md). The sample is
  synthetic and labeled as such; no real specimen or patient data is used.

Sources to study: OpenOrganelle (https://openorganelle.janelia.org/); EMBL LSFM
public datasets (https://www.embl.org/); BioImage Archive
(https://www.ebi.ac.uk/biostudies/bioimages); Zebrafish SPIM atlas data (Nature
Methods papers).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt):

```
4.29 -- Light-Sheet Microscopy Reconstruction
Richardson-Lucy deconvolution (cuFFT, Fourier domain): 32x32 image, PSF sigma=1.60 px, 12 iterations
input  (blurry)   : sum=69.1963  max=0.359447  L2=2.549701
output (deblurred): sum=69.1963  max=0.786365  L2=3.173884
sharpening        : peak x2.1877  L2 x1.2448  (flux ratio 1.0000)
RESULT: PASS (GPU cuFFT matches CPU DFT within rel tol=1.0e-09)
```

The program deconvolves on both the **GPU** (`src/kernels.cu`, cuFFT) and a **CPU
reference** (`src/reference_cpu.cpp`, direct DFT) and asserts they agree. Both run
in double precision and share the per-pixel math, so agreement is ~1e-15 (shown on
stderr); we verify to a 1e-9 relative floor. The `flux ratio 1.0000` confirms RL
conserved total intensity, and `peak x2.19` shows the blur being undone.

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the plane, runs CPU + GPU RL, verifies, reports.
2. [`src/rl_core.h`](src/rl_core.h) — the shared `__host__ __device__` per-pixel RL math.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the cuFFT idea.
4. [`src/kernels.cu`](src/kernels.cu) — the cuFFT RL loop and the element-wise kernels.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted direct-DFT baseline.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, CUDA-event timer, I/O helpers.

## Prior art & further reading

- **BigStitcher** (https://github.com/PreibischLab/BigStitcher) — GPU-accelerated
  LSFM stitching + multi-view fusion; study its FFT-based image correlation and
  multi-GPU deconvolution.
- **DeconvolutionLab2**
  (https://github.com/Biomedical-Imaging-Group/DeconvolutionLab2) — multi-algorithm
  deconvolution; compare RL against Tikhonov/Landweber.
- **CSBDeep / CARE** (https://github.com/CSBDeep/CSBDeep) — deep-learning LSFM
  restoration; the learned complement to model-based deconvolution.
- **Noise2Void** (https://github.com/juglab/n2v) — self-supervised denoising for
  LSFM; a useful pre-step before deconvolution.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Use a CUDA library for the solved sub-problem, hand-write the rest** (PATTERNS.md
§1, the `8.03` cuFFT flagship). cuFFT provides the forward/inverse FFTs;
`complex_mul_scaled`, `ratio_kernel`, and `update_kernel` are one-thread-per-element
kernels for the convolution-theorem multiply and the two RL steps. The full catalog
pattern also lists cuBLAS (view-weight products), custom phase-correlation kernels,
multi-GPU z-plane decomposition, and pinned-memory streaming — all part of the
*full* pipeline described in [THEORY.md](THEORY.md) "real world".

## Exercises

1. **Single precision.** Switch cuFFT to `R2C`/`C2R` (float) and re-measure the
   GPU-vs-CPU error. How much does it grow over 12 iterations, and why (PATTERNS.md
   §4)? Which tolerance would then be honest?
2. **More iterations / stronger blur.** Regenerate with
   `python scripts/make_synthetic.py --sigma 2.5 --iters 40`. Watch the peak keep
   rising — where does RL start amplifying *noise* instead of signal? (This is the
   classic RL over-fitting trade-off.)
3. **Bigger image.** Run `--h 64 --w 64` (or larger) and watch the CPU direct-DFT
   time explode as `O(N²)` while the GPU FFT time barely moves. Plot it.
4. **A non-symmetric PSF.** Replace the Gaussian with an anisotropic PSF (different
   sigma per axis). Confirm the adjoint (`conj(FFT(h))`) still back-projects
   correctly — the symmetric case hid the flip.
5. **Total-variation regularization.** Add a TV penalty to the RL update to suppress
   the noise amplification you saw in exercise 2. Compare edge sharpness.

## Limitations & honesty

- **Reduced scope (CLAUDE.md §13).** This is a **single 2D plane, single view,
  isotropic Gaussian PSF**. Real LSFM is 3D, multi-view (each with an anisotropic
  PSF), tiled, and terabyte-scale — see [THEORY.md](THEORY.md) "Where this sits in
  the real world".
- **Synthetic data.** The sample is generated (`make_synthetic.py`), not a real
  specimen, and is labeled synthetic everywhere. Nothing here implies clinical or
  diagnostic validity.
- **Known PSF.** We assume the PSF is a known Gaussian. Production tools *measure*
  or *blindly estimate* it — a hard problem this version skips.
- **Small image for a readable reference.** The 32×32 size is chosen so the
  `O(N²)` CPU DFT is tractable; the GPU speed-up shown is a **teaching artifact**,
  not a benchmark (CLAUDE.md §12). The real GPU advantage grows with image size.
- **No noise model in the loop.** RL assumes Poisson noise but we do not add a
  regularizer, so many iterations on noisy data will eventually amplify noise
  (exercise 2).
