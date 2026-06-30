# 4.30 — Deconvolution Microscopy

![difficulty](https://img.shields.io/badge/difficulty-Beginner-blue) ![maturity](https://img.shields.io/badge/maturity-Established-informational) ![domain](https://img.shields.io/badge/domain-Medical%20Imaging%20%26%20Image%20Reconstruction-lightgrey)

> **🟢 Beginner · Established** — Domain 4: Medical Imaging & Image Reconstruction · Catalog ID `4.30`
>
> _Educational only — not for clinical use (see CLAUDE.md §8)._

## Summary

A fluorescence microscope never sees a perfectly sharp image: each point of light
is smeared into a small blob described by the **point spread function (PSF)**. The
recorded image is therefore the true specimen *convolved* with the PSF. This
project **deconvolves** — it inverts that blur to recover a sharper estimate of the
specimen — using **Richardson-Lucy (RL)**, the classic iterative deconvolution
algorithm for photon-limited (Poisson) data. Each RL iteration needs two
convolutions; we do them on the GPU with **cuFFT** (the convolution theorem turns a
convolution into a cheap pointwise multiply in the frequency domain) and verify the
result against a plain CPU reference. On a tiny synthetic image the demo restores
five blurred "beads" and shows a ~6× sharpness gain.

## What this computes & why the GPU helps

Wide-field and confocal fluorescence microscopes suffer from out-of-focus blur
described by the point spread function (PSF); iterative deconvolution
(Richardson-Lucy, Landweber) sharpens images by deblurring via the known or
estimated PSF. Each R-L iteration requires two FFT-based convolutions (forward:
estimate × PSF; backward: ratio × PSF_flipped) on volumes as large as 2,048³; GPU
cuFFT reduces each convolution from minutes to seconds.

**The parallel bottleneck:** the two **convolutions per iteration** dominate the
runtime. A direct (spatial) convolution costs `O(N·K)` per image (`N` pixels, `K`
PSF taps); over tens of iterations on a large volume that is enormous. The
**convolution theorem** replaces each one with `FFT → pointwise multiply → inverse
FFT`, i.e. `O(N log N)`, and the FFT is exactly what GPUs (via cuFFT) do fastest.
That is the step we parallelize.

## The algorithm in brief

- **Forward model:** `blurred = convolve(estimate, PSF)` — what the current guess
  would look like through the microscope.
- **Ratio:** `ratio = observed / blurred` (per pixel; guarded against ÷0).
- **Back-projection:** `correction = convolve(ratio, flip(PSF))` — the adjoint.
- **Multiplicative update:** `estimate = estimate × correction` (clamped ≥ 0).
- Repeat for a fixed iteration count. Both convolutions are done with **cuFFT** on
  the GPU; the per-pixel ratio/update are tiny custom kernels.

See [THEORY.md](THEORY.md) for the full science → math → algorithm → GPU-mapping
derivation (including why RL is the maximum-likelihood estimator under Poisson noise
and why we use *circular* convolution so CPU and GPU compute the identical operator).

## Build

Requires **Visual Studio 2026** (v145 toolset) + **CUDA Toolkit 13.3**
(see [docs/BUILD_GUIDE.md](../../../docs/BUILD_GUIDE.md)). This project links the
**cuFFT** library (already wired into the `.vcxproj` and `CMakeLists.txt`).

1. Open `build/deconvolution-microscopy.sln` in Visual Studio 2026.
2. Select the **`Release|x64`** configuration.
3. **Build → Build Solution** (Ctrl+Shift+B). The executable lands in
   `build/x64/Release/deconvolution-microscopy.exe`.

Command-line alternative (Developer PowerShell):

```powershell
msbuild build\deconvolution-microscopy.sln /p:Configuration=Release /p:Platform=x64
```

## Run the demo

```powershell
./demo/run_demo.ps1          # Windows
./demo/run_demo.sh           # Linux/macOS (if CMake build is used)
```

The demo builds if needed, runs on `data/sample/`, deconvolves on both the CPU and
the GPU, prints the result and the GPU-vs-CPU agreement check, and prints a timing
line.

## Data

- **Sample (committed):** `data/sample/blurred_image.txt` — a tiny **synthetic**
  48×48 blurry image so the demo runs with zero downloads.
- **Full dataset:** `scripts/download_data.ps1` / `.sh` point at the EPFL
  deconvolution benchmark, the BioImage Archive, and ImageJ/Fiji samples (each
  under its own license — we do not redistribute).
- **Provenance & license:** see [data/README.md](data/README.md).

Catalog dataset notes: BioImage Archive fluorescence microscopy datasets
(<https://www.ebi.ac.uk/biostudies/bioimages>); EPFL Biomedical Imaging Group
benchmark datasets (<https://bigwww.epfl.ch/deconvolution/>); ImageJ/Fiji sample
datasets (<https://imagej.net/>); COBA microscopy benchmark.

## Expected output

Success looks like [`demo/expected_output.txt`](demo/expected_output.txt): a ~6×
sharpness gain, the recovered bright beads on the diagonal, conserved total
intensity, and `RESULT: PASS`. The program computes the deconvolution on both the
**GPU** (`src/kernels.cu`, cuFFT) and a **CPU reference** (`src/reference_cpu.cpp`,
direct convolution) and asserts they agree to within `atol = 1e-6` — that agreement
is the correctness guarantee. (In practice the worst per-pixel error is ~`1e-13`,
printed on stderr.)

## Code tour

Read in this order:

1. [`src/main.cu`](src/main.cu) — loads the image, builds the PSF, runs CPU + GPU
   RL, verifies, and prints the deterministic report.
2. [`src/reference_cpu.h`](src/reference_cpu.h) — the data model (`Image`, `Psf`)
   and the CPU function contracts.
3. [`src/rl_core.h`](src/rl_core.h) — the shared `__host__ __device__` per-pixel RL
   math (ratio + update), so CPU and GPU compute the *same* arithmetic.
4. [`src/kernels.cuh`](src/kernels.cuh) — the GPU interface + the FFT-convolution idea.
5. [`src/kernels.cu`](src/kernels.cu) — the cuFFT plans, the per-iteration FFT
   pipeline, and the three custom element-wise kernels.
6. [`src/reference_cpu.cpp`](src/reference_cpu.cpp) — the trusted serial baseline
   (direct circular convolution + the RL loop).
7. [`src/util/`](src/util/) — shared `CUDA_CHECK`, the CUDA-event timer, I/O helpers.

## Prior art & further reading

CSBDeep/CARE (<https://github.com/CSBDeep/CSBDeep>) — GPU-accelerated content-aware
restoration *network* (a learned alternative to classical RL); DeconvolutionLab2
(<https://github.com/Biomedical-Imaging-Group/DeconvolutionLab2>) — a multi-algorithm
deconvolution toolbox (RL, Landweber, Tikhonov, TV) worth studying for the algorithm
zoo; FlowDec (<https://github.com/hammerlab/flowdec>) — a TensorFlow FFT-based GPU RL
implementation (compare its FFT-convolution structure to ours); N2V
(<https://github.com/juglab/n2v>) — self-supervised denoising often applied *before*
deconvolution.

Study these to learn the production approach; **do not copy code wholesale** —
reimplement didactically and credit the source (CLAUDE.md §2).

## CUDA pattern used here

**Use cuFFT for FFT convolution** (PATTERNS.md §1, the same pattern as flagship
`8.03`). cuFFT does the heavy real-to-complex 2-D FFTs; the only custom kernels are
the element-wise complex multiply (apply the PSF / its adjoint in frequency, with
the `1/N` normalization folded in) and the per-pixel RL ratio/update. The PSF is
transformed **once** and its spectrum reused every iteration (and conjugated on the
fly for the adjoint step). The original catalog also lists 3-D in-place FFTs,
texture memory for the PSF, batched multi-channel FFTs, and pinned-memory streaming
— all described in THEORY.md "Where this sits in the real world".

## Exercises

1. **More iterations.** Raise `RL_ITERS` in `src/main.cu` (e.g. 100). Watch the
   sharpness climb — and then watch *noise* amplify. RL has no built-in stopping
   rule; where would you stop?
2. **Wrong PSF.** Deconvolve with a `sigma` that does not match the blur (change
   `PSF_SIGMA` in `main.cu` but not in `make_synthetic.py`). Observe ringing /
   artifacts — real microscopy is acutely sensitive to PSF accuracy.
3. **Total-variation regularization.** Add a TV penalty to the update to suppress
   the noise amplification from exercise 1 (DeconvolutionLab2 implements this).
4. **Bigger images.** `python scripts/make_synthetic.py --w 256 --h 256` and
   re-time. How does the GPU-vs-CPU gap grow? (It should widen — `O(N log N)` vs
   `O(N·K)`.)
5. **Single precision.** Switch the cuFFT path to `CUFFT_R2C`/`CUFFT_C2R` (float)
   and measure the speed gain vs the accuracy loss. What tolerance still passes?

## Limitations & honesty

- **Teaching-scale, 2-D.** Real deconvolution is **3-D** (z-stacks) on volumes up
  to 2,048³; we use a tiny 2-D image so the demo is instant and the math is
  legible. The 3-D extension is "same pipeline, 3-D cuFFT plan" — described in
  THEORY.md, not implemented here.
- **Synthetic data, known PSF.** The sample is **synthetic** (no real specimen) and
  we *assume the PSF is known and exact*. Real workflows **measure** the PSF from
  fluorescent beads, or do **blind** deconvolution (jointly estimating the PSF) —
  both out of scope here.
- **Circular convolution.** We use periodic (wrap-around) convolution because it is
  exactly what an FFT computes, which lets us verify GPU == CPU. Real images are not
  periodic; production code pads/apodizes the borders to avoid wrap-around
  artifacts (THEORY.md "Numerical considerations").
- **No noise model beyond the ratio.** RL implicitly assumes Poisson noise; we add
  no explicit denoising. The "noise amplification" exercise shows why that matters.
- This is **study material, not a clinical tool.** No output here is validated for
  any diagnostic or therapeutic use.
