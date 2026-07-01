# 4.8 — Deformable Image Registration

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.8`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

**Deformable image registration (DIR)** aligns two images that show the same anatomy in different shapes — a
lung at inhale vs. exhale, a brain across two scans, a patient before vs. during radiotherapy. Instead of one
global rotation+shift, DIR estimates a *dense displacement vector field* (DVF): a little arrow at **every
pixel** saying where it should move so the **moving** image lines up with the **fixed** one. This project
implements the classic **Thirion's Demons** algorithm in 2-D on the GPU. Each iteration warps the moving
image, computes a per-pixel force from the intensity mismatch and image gradient, and Gaussian-smooths the
field for regularization — three fully data-parallel passes that map cleanly onto one GPU thread per pixel.
On the committed synthetic sample it drives the image dissimilarity (SSD) down by **99.9%** while the GPU
result matches an independent CPU reference to ~5e-15 pixels.

## What this computes & why the GPU helps

Deformable image registration (DIR) estimates a dense displacement vector field (DVF) that maps a moving
image to a fixed image, minimizing an image dissimilarity metric (NCC, NMI, SSD) subject to a regularization
penalty (bending energy, diffusion). Classical optimization (Demons, B-spline free-form deformation) requires
hundreds of gradient-descent iterations on each voxel of a dense DVF — ~10⁹ parameters for a 256³ volume —
making per-iteration GPU parallelism essential. Learning-based methods (VoxelMorph) infer the DVF in a single
forward pass (<1 s GPU vs. 2+ hrs ANTs CPU). LDDMM adds geodesic shooting on the diffeomorphism group,
computable via GPU-accelerated Fourier-domain operators.

**The parallel bottleneck:** every Demons iteration touches **every pixel three times** — a warp (bilinear
gather), a force computation, and a separable Gaussian smoothing (two stencil passes). For a real 256³ volume
that is ~10⁷ voxels × hundreds of iterations ≈ 10⁹–10¹⁰ independent per-voxel updates. Those updates have no
inter-pixel dependency within a pass, so the GPU runs them all at once — one thread per pixel — turning an
hours-long CPU job into seconds. In this demo the GPU already beats the CPU ~9× on a tiny 64×64 image; the
gap widens with size.

## The algorithm in brief

- **Warp** the moving image by the current DVF: `Mw(x) = M(x + u(x))` via **bilinear interpolation** (a gather).
- **Force** per pixel (Thirion's optical-flow demon force):
  `du = (F − Mw)·∇F / (|∇F|² + (F − Mw)² + ε)`, then `u += du`.
- **Regularize** by **Gaussian-smoothing** the whole displacement field (a diffusion prior that keeps the
  deformation spatially coherent) — done as a **separable** X-pass then Y-pass.
- **Iterate** ~100× until the warped moving image matches the fixed image (SSD stops dropping).

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)).

1. Open `build/deformable-image-registration.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/deformable-image-registration.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\deformable-image-registration.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, prints the SSD-before/after result, shows the GPU-vs-CPU
agreement check, and prints a timing line.

## Data

- **Sample (committed):** `data/sample/dir_pair.txt` — a tiny, **synthetic** fixed/moving image pair (a
  Gaussian blob shifted + stretched) so the demo runs with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print pointers to real registration benchmarks
  (they never bypass any dataset license/registration).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: OASIS brain MRI (https://www.oasis-brains.org/) — used in Learn2Reg challenge;
Learn2Reg 2022 challenge (https://learn2reg.grand-challenge.org/) — lung, brain, abdominal; DIR-Lab lung CT
deformation dataset (https://dir-lab.com/); 4D-CT lung datasets for respiratory motion.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt):

```
4.8 -- Deformable Image Registration
Demons DIR: 64x64 image, 120 iters, sigma=1.50 px (radius=5)
SSD before = 51.8191
SSD after  = 0.0643  (99.88% reduction)
mean |displacement| = 4.5079 px
u_x along center row (8 samples): 1.3479 2.2833 3.3400 4.3975 5.3793 6.4929 8.0949 10.5250
RESULT: PASS (GPU field matches CPU within tol=1.0e-03 px)
```

The program runs Demons on both the **GPU** (`src/kernels.cu`) and a **CPU reference**
(`src/reference_cpu.cpp`); the two share their per-pixel math via `src/demons.h`, so their displacement
fields agree to ~5e-15 px (far under the 1e-3 px tolerance). The 99.9% SSD drop is the science check: the
moving blob really did snap onto the fixed one.

## Code tour

Read in this order:

1. [`src/demons.h`](src/demons.h) — the shared `__host__ __device__` per-pixel physics (warp, gradient,
   Thirion force, Gaussian). **Start here** — everything else calls these.
2. [`src/main.cu`](src/main.cu) — loads the image pair, runs CPU + GPU, verifies, reports SSD/displacement.
3. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the three-pass thread-mapping idea.
4. [`src/kernels.cu`](src/kernels.cu) — the three kernels and the ping-pong iteration loop.
5. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline (a direct mirror of the GPU).
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **VoxelMorph** (https://github.com/voxelmorph/voxelmorph) — TF/PyTorch *unsupervised* CNN that predicts the
  DVF in one forward pass; study how learning replaces the iterative optimizer.
- **Plastimatch** (https://plastimatch.org/) — production **GPU B-spline and Demons** with DICOM-RT support;
  the closest real-world sibling of this project — see how they handle 3-D, multi-resolution, and masks.
- **ANTs** (https://github.com/ANTsX/ANTs) — gold-standard **SyN** symmetric diffeomorphic registration
  (CPU, used to generate ground truth); study the diffeomorphism guarantee this teaching version lacks.
- **TransMorph** (https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration) —
  Swin-transformer DIR; a look at where the field is heading.

Study these to learn the production approach; **do not copy code wholesale** — reimplement didactically and
credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Per-pixel gather + separable stencil, iterated with ping-pong buffers.** Each Demons iteration is three
data-parallel kernels over the image (one thread per pixel, a 16×16 block over a 2-D grid): a **warp** that
bilinearly gathers the moving image (the 2-D sibling of the catalog's "custom trilinear interpolation kernel
for warp"), and a **separable Gaussian** smoothing done as an X-pass then a Y-pass writing into a second
buffer (double-buffered stencil). The DVF and both images live in GPU global memory for the whole loop, so
no host↔device traffic happens between iterations. This reduced-scope version does *not* use cuFFT/cuBLAS
(those belong to the LDDMM / Hessian variants described in THEORY §7); a shared-memory-tiled Gaussian is left
as an exercise.

## Exercises

1. **Shared-memory Gaussian.** The `gauss_x/gauss_y` kernels re-read each pixel's neighbourhood from global
   memory. Tile a block's rows/columns into shared memory (with a halo) and measure the speed-up — this is
   the classic stencil optimization from project 6.04/7.10.
2. **Multi-resolution (pyramid) Demons.** Downsample F and M by 2× and 4×, register coarse→fine, upsampling
   the DVF between levels. This captures large motions that the single-scale version misses and converges in
   far fewer fine-level iterations.
3. **Normalized cross-correlation (NCC) force.** SSD assumes identical intensity scales. Swap the force for a
   local-NCC metric (the catalog names it) so the method survives brightness/contrast differences between
   scans.
4. **Precompute ∇F once.** The fixed-image gradient never changes; hoist it out of the per-iteration force
   kernel into a one-time buffer and measure the reduction in global reads.
5. **Jacobian / folding check.** Compute `det(I + ∇u)` across the field; where it goes ≤ 0 the deformation
   *folds* (non-physical). Report the min Jacobian — this is how a real tool flags a bad registration.

## Limitations & honesty

- **Synthetic data.** The sample is a Gaussian blob, not anatomy — chosen so the answer is unambiguous and the
  gradient is defined everywhere. Labeled synthetic throughout; **not** patient-derived.
- **2-D, single-resolution, SSD only.** Real DIR is 3-D, multi-resolution, and uses NCC/NMI to tolerate
  intensity differences between modalities/scans. See THEORY §7.
- **Not diffeomorphic.** Plain additive Demons can produce a folding (non-invertible) field for large
  motions; *diffeomorphic* Demons and SyN add that guarantee. Here the motion is small enough to stay well-behaved.
- **Regularization is a simple Gaussian diffusion prior**, not a bending-energy / elastic model.
- **Teaching timings, not benchmarks.** The ms figures illustrate the GPU-vs-CPU gap on a tiny image; they are
  not a performance claim (CLAUDE.md §12). **Not for any clinical use.**
