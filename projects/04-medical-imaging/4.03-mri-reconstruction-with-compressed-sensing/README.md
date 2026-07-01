# 4.3 — MRI Reconstruction with Compressed Sensing

![difficulty](https://img.shields.io/badge/difficulty-Intermediate-blue) ![maturity](https://img.shields.io/badge/maturity-Active%20R%26D-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟡 Intermediate · Active R&D** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.3`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

An MRI scanner does not photograph anatomy — it measures **k-space**, samples of
the image's 2D Fourier transform, one line at a time, which makes a full scan slow.
**Compressed sensing (CS)** skips most of those lines to scan several times faster,
then *reconstructs* the missing information by exploiting the fact that medical
images are compressible (sparse in some transform domain). This project implements
a complete, honest CS-MRI reconstruction for one Cartesian slice: it under-samples
k-space, reconstructs the image with **FISTA** (an accelerated proximal-gradient
solver whose two FFTs per iteration run on **cuFFT**), and proves — against a known
ground truth — that CS recovers the image far better than the naive "zero-filled"
reconstruction. The GPU result is checked against a plain, obviously-correct CPU
implementation of the exact same math.

## What this computes & why the GPU helps

MRI acquires k-space (Fourier-domain) samples; compressed sensing (CS) reconstructs images from highly under-sampled k-space using sparsity priors (wavelet, total variation), enabling 4–8× scan acceleration. The core computation is a sequence of FFTs (here Cartesian; NUFFT/NFFT for arbitrary trajectories), followed by iterative soft-thresholding / proximal operators. At clinical resolution (~256³) each iteration is ~10⁹ operations; GPU parallelism reduces each FFT to milliseconds vs. seconds on CPU. Multi-channel parallel imaging (SENSE, GRAPPA, PICS) adds per-coil FFTs (~32 channels), multiplying the compute by the coil count and making the GPU essential.

**The parallel bottleneck:** each FISTA iteration performs a **forward FFT** and an
**inverse FFT** of the whole image (plus cheap per-pixel updates). The two FFTs
dominate — they are `O(N log N)` over every voxel of every coil, repeated for tens
of iterations. That is exactly what cuFFT accelerates: this project hands both
transforms to `cufftExecC2C` and keeps only the tiny masking / soft-threshold /
momentum steps as custom one-thread-per-pixel kernels.

## The algorithm in brief

- **Forward model** `E = M∘F`: 2D FFT of the image, then keep only sampled k-space.
- **Objective**: `min_x  ½‖M F x − y‖² + λ‖x‖₁` (data fit + L1 sparsity).
- **FISTA** (Fast Iterative Shrinkage-Thresholding): proximal-gradient with Nesterov
  momentum — `O(1/k²)` convergence.
  - gradient of the data term: `∇ = F⁻¹{ M (F z − y) }`
  - proximal step: `x = softThreshold(z − ∇, λ)`  (the L1 prox = shrinkage)
  - momentum: `z = x + β(x − x_prev)`
- **Verification**: GPU (cuFFT) image vs CPU (hand radix-2 FFT) image, and CS error
  vs the zero-filled baseline against the synthetic ground truth.
- Catalog also lists SENSE, GRAPPA, NUFFT, PICS, ADMM/Split-Bregman, TV/wavelet
  sparsity, k-t SENSE — how each slots into this same loop is in `THEORY.md`.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation.

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)). This project links
**cuFFT** (`cufft.lib`) for the 2D FFTs — the `.vcxproj` already declares it.

1. Open `build/mri-reconstruction-with-compressed-sensing.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/mri-reconstruction-with-compressed-sensing.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\mri-reconstruction-with-compressed-sensing.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (CMake build; links CUDA::cufft)
```

The demo builds if needed, runs on `data/sample/kspace_sample.txt`, prints the
deterministic result, shows the GPU-vs-CPU agreement check, and prints a timing
line on stderr.

## Data

- **Sample (committed):** `data/sample/kspace_sample.txt` — a tiny **synthetic**
  32×32 under-sampled k-space slice + mask + ground truth, so the demo runs offline
  with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` print how to obtain the real
  (credentialed) datasets; they never bypass any registration.
- **Provenance & license:** see [data/README.md](data/README.md).

Real datasets (all require a data-use agreement): **fastMRI** (NYU/Meta), knee +
brain raw k-space; **Calgary-Campinas-359**, multi-channel brain MRI k-space;
**SKM-TEA** (Stanford knee MRI).

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt):

```
image: 32x32   sampled k-space: 348/1024 (34.0%)   lambda=0.0100   iters=60
error vs truth (RMS): zero-filled=0.051109  CS-reconstructed=0.005779
CS improvement: 8.84x lower error than zero-filling
RESULT: PASS (GPU cuFFT recon matches CPU FISTA within tol; CS beats zero-fill)
```

The program reconstructs on both the **GPU** (`src/kernels.cu`, cuFFT) and a **CPU
reference** (`src/reference_cpu.cpp`, hand radix-2 FFT) and asserts they agree
within a documented tolerance (observed RMS difference ≈ `3e-8` — see
`THEORY.md` "How we verify correctness"). It also asserts CS beats the zero-filled
baseline, which is the science.

## Code tour

Read in this order:

1. [`src/cs_core.h`](src/cs_core.h) — the shared `__host__ __device__` per-pixel
   math (complex ops, the L1 soft-threshold, the data-consistency residual). Both
   the CPU and GPU paths call these *same* functions.
2. [`src/main.cu`](src/main.cu) — loads k-space, runs CPU + GPU FISTA, verifies, reports.
3. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted radix-2 FFT and the
   readable FISTA loop.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the thread mapping.
5. [`src/kernels.cu`](src/kernels.cu) — the cuFFT calls (documented, not a black box)
   and the per-pixel kernels.
6. [`src/util/`](src/util/) — shared `CUDA_CHECK`, event timer, I/O helpers.

## Prior art & further reading

- **[BART](https://github.com/mrirecon/bart)** (Berkeley Advanced Reconstruction
  Toolbox) — production CS-MRI: GPU-accelerated PICS, SENSE, NUFFT. Study its
  `pics` command and its operator/proximal abstractions.
- **[SigPy](https://github.com/mikgroup/sigpy)** — Python GPU (CuPy) MRI signal
  processing and NUFFT; a clean, readable reference for the linear operators.
- **[MIRT](https://github.com/JeffFessler/MIRT.jl)** (Michigan Image Reconstruction
  Toolbox) — Julia/MATLAB iterative reconstruction; excellent for the math.
- **[PyNUFFT](https://github.com/jyhmiinlin/pynufft)** — NUFFT with CUDA/OpenCL
  backends; study for the non-Cartesian gridding this project simplifies away.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Use a CUDA library without it being a black box** (PATTERNS.md §1 "FFT" row,
exemplar `8.03`): the expensive, solved part — the 2D FFT — is handed to **cuFFT**
(`cufftPlan2d` + `cufftExecC2C`), with `kernels.cu` documenting exactly what it
computes, the data layout it expects, and what hand-rolling would take. The cheap
per-pixel steps (mask the residual, soft-threshold, momentum extrapolation) are
three tiny **one-thread-per-pixel** kernels whose arithmetic is shared verbatim with
the CPU reference via `cs_core.h` — so CPU and GPU differ *only* in the FFT engine.

## Exercises

1. **Change the acceleration.** Regenerate the sample at a harsher under-sampling
   (`python scripts/make_synthetic.py --keep 0.20`). At what point does FISTA stop
   recovering the blobs? Watch the "CS improvement" factor.
2. **Sweep λ.** Try `--lam 0.0` (no sparsity → you just fit noise/aliasing) and a
   large `--lam`. Plot reconstruction error vs λ; you will find the classic
   bias/variance "L-curve" sweet spot.
3. **Add total-variation sparsity.** Replace the identity `Psi` with a
   finite-difference gradient (soft-threshold `∇x` instead of `x`). This is the more
   realistic MRI prior — sketch the extra kernels needed (see `THEORY.md`).
4. **Two coils (SENSE).** Extend `KSpaceData` and the loop to two receive coils with
   sensitivity maps; the data term becomes `Σ_c ‖M F S_c x − y_c‖²`. cuFFT batches
   the per-coil FFTs in one call.
5. **Precision study.** Switch `Cplx` and the FFT to double precision (cuFFT
   `Z2Z`). How much does the GPU-vs-CPU agreement tighten, and at what speed cost?

## Limitations & honesty

- **Teaching-scale, reduced scope.** One 32×32 **Cartesian** slice, **single coil**,
  **identity** sparsity, and a **synthetic** phantom. Production CS-MRI is 3D,
  multi-coil (SENSE/GRAPPA/PICS), often **non-Cartesian** (needing a NUFFT with
  gridding — described in `THEORY.md` but not implemented here), with wavelet/TV
  priors and ADMM/Split-Bregman solvers.
- **The data is synthetic and labeled synthetic everywhere.** It carries a known
  ground truth *purely* so the demo can prove CS works; this implies **no clinical
  validity**. Do not use any output for diagnosis or treatment.
- **Timing is a teaching artifact, not a benchmark.** On this tiny slice the GPU is
  *slower* than the CPU (the two per-iteration FFTs are launch-bound); the GPU wins
  only at clinical sizes. See PATTERNS.md §7.
- **Verification tolerance is honest.** The GPU (cuFFT) and CPU (radix-2) FFTs round
  differently, so we verify to a small, physically-negligible tolerance (§ "How we
  verify" in THEORY), not bit-identity — though observed agreement is ~`3e-8`.
