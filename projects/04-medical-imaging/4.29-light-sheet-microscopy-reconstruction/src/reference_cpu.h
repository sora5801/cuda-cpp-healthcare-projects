// ===========================================================================
// src/reference_cpu.h  --  Image model + CPU Richardson-Lucy reference
// ---------------------------------------------------------------------------
// Project 4.29 : Light-Sheet Microscopy Reconstruction
//                (reduced-scope teaching version -- see THEORY.md "real world").
//
// WHAT THIS PROJECT COMPUTES
//   Light-sheet fluorescence microscopy (LSFM) records optically-sectioned 3D
//   stacks that are BLURRED by the microscope's point-spread function (PSF) and
//   corrupted by photon (Poisson) noise. Reconstruction DEBLURS them, classically
//   with multi-view Richardson-Lucy (RL) deconvolution. This project implements
//   the didactic core of that pipeline: RL deconvolution of a single 2D plane
//   with a Gaussian PSF. The GPU does the two convolutions per iteration in the
//   FOURIER domain with cuFFT (the catalog's exact pattern); the CPU reference
//   does the same convolutions with a direct DFT so we can verify.
//
// WHY A GPU / cuFFT
//   Each RL iteration is two convolutions of the whole volume; a 1000^3 stack is
//   ~10^12 multiply-accumulates per iteration, over dozens of iterations and
//   several views. Convolution is a pointwise MULTIPLY in Fourier space, so the
//   FFT turns an O(N^2)-per-pixel spatial convolution into O(N log N) total --
//   and cuFFT runs that FFT on the GPU. This teaching version keeps the image
//   tiny (so the naive CPU DFT is tractable) but uses the identical algorithm.
//
// THE DATA FORMAT (data/README.md has the full spec)
//   A whitespace/newline text file:
//     line 1:  H W  SIGMA  ITERS            (image height, width; PSF sigma px; RL iters)
//     then  H*W  floats: the BLURRY, noisy measured image  b  (row-major).
//   The committed sample is SYNTHETIC (a known ground-truth image blurred by a
//   Gaussian and given Poisson-like noise) so RL has a real target to recover.
//
//   This header is PURE C++ (no CUDA) so cl.exe can compile reference_cpu.cpp.
//   kernels.cu re-uses LsfmData. Per-pixel RL math lives in rl_core.h (shared).
//
// READ THIS AFTER: rl_core.h. READ THIS BEFORE: kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// LsfmData: one measured 2D plane plus the reconstruction parameters.
//   Stored row-major: pixel (row r, col c) is at index  r*W + c.
//   `measured` holds the blurry+noisy image b that we deconvolve.
// -----------------------------------------------------------------------------
struct LsfmData {
    int H = 0;                    // image height  (rows)
    int W = 0;                    // image width   (cols)
    double sigma = 0.0;           // Gaussian PSF standard deviation, in pixels
    int iters = 0;                // number of Richardson-Lucy iterations to run
    std::vector<double> measured; // [H*W] observed blurry, noisy image (row-major)
};

// Load an LsfmData from the text format documented above / in data/README.md.
//   Throws std::runtime_error on a missing file or a malformed header so demos
//   fail loudly instead of silently running on garbage.
LsfmData load_lsfm(const std::string& path);

// -----------------------------------------------------------------------------
// gaussian_psf: build a normalized (sum == 1) Gaussian PSF of size H x W,
//   CENTERED AT PIXEL (0,0) with wrap-around (FFT convention), sigma in pixels.
//   Centering at the origin (not the image center) means convolving with it does
//   NOT shift the image -- crucial so the CPU DFT convolution and the cuFFT
//   convolution produce the same, un-shifted result. Returns a row-major [H*W].
//   Used by BOTH the CPU reference and (copied to the GPU) the kernels.
// -----------------------------------------------------------------------------
std::vector<double> gaussian_psf(int H, int W, double sigma);

// -----------------------------------------------------------------------------
// deconvolve_cpu: the trusted CPU Richardson-Lucy reference.
//   Runs `d.iters` RL iterations on d.measured with the Gaussian PSF and returns
//   the deblurred estimate (size H*W, row-major). Uses a direct DFT-based
//   CIRCULAR convolution -- mathematically identical to what cuFFT computes on
//   the GPU -- and the shared rl_core.h per-pixel update, so GPU and CPU agree.
//   This is slow (O((H*W)^2) per convolution) BUT OBVIOUSLY CORRECT: the whole
//   point of a reference. See reference_cpu.cpp for the heavily-commented body.
// -----------------------------------------------------------------------------
void deconvolve_cpu(const LsfmData& d, std::vector<double>& estimate);

// -----------------------------------------------------------------------------
// image_stats: reduce an image to three deterministic, order-independent
//   summary numbers used both to REPORT the result and to VERIFY GPU vs CPU:
//     out_sum   = sum of all pixels        (RL conserves total intensity ~ flux)
//     out_max   = maximum pixel value      (peak brightness; sharpens with iters)
//     out_l2    = sqrt(sum of pixel^2)     (energy; grows as blur is undone)
//   Summed in a fixed left-to-right order so the value is bit-reproducible.
// -----------------------------------------------------------------------------
void image_stats(const std::vector<double>& img, double& out_sum,
                 double& out_max, double& out_l2);
