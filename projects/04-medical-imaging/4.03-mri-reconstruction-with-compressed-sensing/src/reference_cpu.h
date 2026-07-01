// ===========================================================================
// src/reference_cpu.h  --  MRI/CS data model, CPU FFT, and CPU FISTA reference
// ---------------------------------------------------------------------------
// Project 4.3 : MRI Reconstruction with Compressed Sensing
//
// WHAT THIS PROJECT COMPUTES
//   An MRI scanner does NOT measure an image. It measures k-space: samples of the
//   image's 2D Fourier transform. A full scan samples every k-space line, which is
//   slow. COMPRESSED SENSING skips most of the lines (say, keep 1 in 3) to scan
//   4-8x faster, then RECONSTRUCTS the missing information by exploiting the fact
//   that medical images are compressible (sparse in a transform domain). We solve
//
//        minimize_x  (1/2)|| M F x - y ||_2^2  +  lambda || Psi x ||_1
//
//   where F is the 2D FFT, M is the sampling mask, y is the measured under-sampled
//   k-space, Psi is a sparsifying transform, and lambda trades data fit against
//   sparsity. We minimize it with FISTA (an accelerated proximal-gradient method).
//
// WHY A GPU (the catalog's "cuFFT for gridded FFT" pattern)
//   Every FISTA iteration needs a forward FFT and an inverse FFT of the whole
//   image; clinical volumes are ~256^3 with ~32 receive coils, so each iteration is
//   ~10^9 operations and a scan needs tens of iterations -- seconds on a CPU,
//   milliseconds on a GPU with cuFFT. This teaching project uses a single 2D slice
//   and a single coil so the whole thing runs offline in a fraction of a second,
//   but the STRUCTURE (masked-FFT forward op + proximal iteration + cuFFT) is
//   exactly the production one. See THEORY "GPU mapping".
//
//   The GPU path (kernels.cu) uses cuFFT for the FFTs; this CPU reference uses a
//   plain, obviously-correct radix-2 FFT so the two can be cross-checked.
//
//   Pure C++ header (no CUDA). kernels.cu reuses these structs and cs_core.h.
//
// READ THIS BEFORE: reference_cpu.cpp, main.cu.  READ cs_core.h FIRST.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "cs_core.h"   // Cplx and the shared __host__ __device__ per-element math

// ---------------------------------------------------------------------------
// KSpaceData: everything the reconstructor needs, loaded from the sample file.
//   The image is n x n (n a power of two so the radix-2 FFT is simple). Arrays are
//   row-major: index (row r, col c) -> r*n + c.
// ---------------------------------------------------------------------------
struct KSpaceData {
    int n = 0;                     // image side length (power of two); grid is n x n
    float lambda = 0.0f;           // L1 regularization weight (sparsity strength)
    int iters = 0;                 // number of FISTA iterations to run
    std::vector<Cplx> kspace;      // [n*n] measured, ALREADY zero-filled where unsampled
    std::vector<int>  mask;        // [n*n] 1 = k-space position acquired, 0 = skipped
    std::vector<float> truth;      // [n*n] OPTIONAL ground-truth magnitude image (synthetic)
    bool has_truth = false;        // true if the sample carried a ground-truth image
};

// load_kspace: parse the text sample (format documented in data/README.md):
//   line 1: "<n> <lambda> <iters> <has_truth>"
//   then n*n lines "re im mask [truth]" in row-major order.
// Throws std::runtime_error on a malformed/absent file so demos fail loudly.
KSpaceData load_kspace(const std::string& path);

// ---------------------------------------------------------------------------
// fft2_cpu / ifft2_cpu: the trusted CPU reference 2D FFT and its inverse.
//   Implemented as a separable radix-2 Cooley-Tukey FFT (rows then columns) in
//   DOUBLE precision internally for accuracy, writing back into a Cplx (float)
//   buffer. `fft2_cpu` gives the UN-normalized forward transform (matches cuFFT's
//   convention); ifft2_cpu applies the 1/(n*n) normalization so that
//   ifft2(fft2(x)) == x. These define the operators F and F^{-1} in the math above.
// ---------------------------------------------------------------------------
void fft2_cpu(std::vector<Cplx>& data, int n);        // in-place forward  (no 1/N)
void ifft2_cpu(std::vector<Cplx>& data, int n);       // in-place inverse  (with 1/(n*n))

// ---------------------------------------------------------------------------
// reconstruct_cpu: the full CPU CS-MRI reconstruction (FISTA).
//   Runs `d.iters` FISTA iterations on the CPU and returns the reconstructed
//   MAGNITUDE image (size n*n). This is the ground truth the GPU result is checked
//   against. The algorithm is spelled out step-by-step in reference_cpu.cpp and in
//   THEORY "the algorithm".
//     * d       : the loaded problem (k-space, mask, lambda, iters)
//     * out_mag : filled with the |x| magnitude image (size n*n), row-major
// ---------------------------------------------------------------------------
void reconstruct_cpu(const KSpaceData& d, std::vector<float>& out_mag);

// ---------------------------------------------------------------------------
// zero_filled_magnitude: the naive baseline reconstruction -- just inverse-FFT the
//   zero-filled k-space with NO regularization. This is what you get "for free" and
//   it is visibly corrupted by aliasing artifacts; CS is what removes them. main.cu
//   reports its error against the truth so the learner sees the improvement.
// ---------------------------------------------------------------------------
void zero_filled_magnitude(const KSpaceData& d, std::vector<float>& out_mag);
