// ===========================================================================
// src/dft_core.h  --  Shared (host + device) 2-D DFT / iDFT for the
//                     data-consistency step
// ---------------------------------------------------------------------------
// Project 4.4 : Deep-Learning MRI/CT Reconstruction  (REDUCED-SCOPE TEACHING VERSION)
//
// WHAT THIS IS
//   The data-consistency (DC) layer of an unrolled reconstruction has to move
//   between the IMAGE domain and the FREQUENCY (k-space) domain, because MRI
//   MEASURES k-space, not the image. Real pipelines use cuFFT for this (an
//   O(N log N) FFT); we deliberately HAND-ROLL a direct O(N^2) 2-D DFT instead,
//   for two teaching reasons:
//     1. NO BLACK BOX. A learner can read every multiply-add of the transform
//        here, then graduate to cuFFT knowing exactly what it computes.
//     2. The DFT is EMBARRASSINGLY PARALLEL per output frequency -- each output
//        k-space sample is an independent reduction over all image pixels -- so
//        it is a clean second GPU pattern (a "gather + reduce per output").
//   For a real 256x256 image the direct DFT is far too slow; that is precisely
//   why production uses cuFFT. We use a TINY image so O(N^2) is instant, and we
//   say so loudly in THEORY.md.
//
//   A 2-D image of N = ny*nx real pixels has N complex frequencies. We store
//   complex numbers as two parallel float arrays (re[], im[]) -- "structure of
//   arrays" -- so GPU threads reading consecutive frequencies get coalesced
//   loads. The transform pair (unnormalized forward, 1/N-normalized inverse):
//
//     Forward  F[v,u] = sum_{y,x} img[y,x] * exp(-2pi i (v*y/ny + u*x/nx))
//     Inverse  img[y,x] = (1/N) sum_{v,u} F[v,u] * exp(+2pi i (v*y/ny + u*x/nx))
//
//   iDFT(DFT(img)) == img (to floating-point precision), which we exploit: the
//   DC step is DFT -> overwrite sampled bins -> iDFT.
//
//   All angle/twiddle arithmetic lives in ONE __host__ __device__ pair of
//   functions so the CPU reference and GPU kernel compute the SAME sums.
//
// READ THIS AFTER: recon_core.h. READ BEFORE: kernels.cu, reference_cpu.cpp.
// ===========================================================================
#pragma once

#include <cstddef>
#include <cmath>     // sinf/cosf on host; device uses the CUDA math intrinsics

#ifdef __CUDACC__
#define DFT_HD __host__ __device__
#else
#define DFT_HD
#endif

// 2*pi as a float constant (used to build the DFT twiddle angles).
#ifndef DFT_TWO_PI
#define DFT_TWO_PI 6.28318530717958647692f
#endif

// Flat row-major index of a (y,x) sample in an (ny x nx) grid. Used for both the
// image and the k-space arrays (both are ny x nx). Identical to img_idx in
// recon_core.h but duplicated here so this header stands alone.
DFT_HD inline std::size_t kidx(int y, int x, int nx) {
    return static_cast<std::size_t>(y) * nx + x;
}

// ---------------------------------------------------------------------------
// dft_forward_pixel: compute ONE output frequency F[v,u] of the forward 2-D DFT.
//   This is a full reduction over every input pixel -- the per-output work of a
//   "gather + reduce". The CPU reference calls it for all (v,u); the GPU kernel
//   assigns one thread per (v,u). Both accumulate in float in the SAME order
//   (y outer, x inner), so the sums match to FMA precision.
//
//   Params:
//     img      : [ny*nx] real image (row-major).
//     v,u      : the output frequency indices (0..ny-1, 0..nx-1).
//     ny,nx    : dimensions.
//     out_re, out_im : the real/imag parts of F[v,u] (written through pointers).
//   Complexity: O(ny*nx) per output -> O((ny*nx)^2) for the whole transform.
// ---------------------------------------------------------------------------
DFT_HD inline void dft_forward_pixel(const float* img, int v, int u,
                                     int ny, int nx, float* out_re, float* out_im) {
    float re = 0.0f, im = 0.0f;             // accumulators for this frequency bin
    for (int y = 0; y < ny; ++y) {
        // Precompute the y-part of the phase once per row (an honest micro-opt
        // that does NOT change the arithmetic order across host/device).
        const float phase_y = DFT_TWO_PI * (static_cast<float>(v) * y / ny);
        for (int x = 0; x < nx; ++x) {
            // Total phase for pixel (y,x) at frequency (v,u); minus sign = forward.
            const float ang = phase_y + DFT_TWO_PI * (static_cast<float>(u) * x / nx);
            const float c = cosf(ang);      // real part of exp(-i ang) is cos(ang)
            const float s = sinf(ang);      // exp(-i ang) = cos(ang) - i sin(ang)
            const float p = img[kidx(y, x, nx)];
            re += p * c;                    // accumulate real part
            im -= p * s;                    // minus sin -> forward transform sign
        }
    }
    *out_re = re;
    *out_im = im;
}

// ---------------------------------------------------------------------------
// idft_pixel: compute ONE output image pixel img[y,x] of the inverse 2-D DFT.
//   Symmetric to the forward transform: reduce over all frequencies, +i sign,
//   and divide by N = ny*nx. The image is real, so we return only the real part
//   (the imaginary part is ~0 to FMA precision for a real-valued original).
//
//   Params:
//     fre,fim : [ny*nx] real/imag parts of the k-space data (row-major).
//     y,x     : the output pixel this call reconstructs.
//     ny,nx   : dimensions.
//   Returns: the real part of the inverse transform at (y,x).
// ---------------------------------------------------------------------------
DFT_HD inline float idft_pixel(const float* fre, const float* fim,
                               int y, int x, int ny, int nx) {
    float re = 0.0f;                        // accumulate the real part of the sum
    for (int v = 0; v < ny; ++v) {
        const float phase_v = DFT_TWO_PI * (static_cast<float>(v) * y / ny);
        for (int u = 0; u < nx; ++u) {
            const float ang = phase_v + DFT_TWO_PI * (static_cast<float>(u) * x / nx);
            const float c = cosf(ang);      // +i sign: exp(+i ang) = cos + i sin
            const float s = sinf(ang);
            const std::size_t k = kidx(v, u, nx);
            // Re{ (fre + i fim)(cos + i sin) } = fre*cos - fim*sin.
            re += fre[k] * c - fim[k] * s;
        }
    }
    const float N = static_cast<float>(ny) * static_cast<float>(nx);
    return re / N;                          // inverse DFT normalization (1/N)
}
