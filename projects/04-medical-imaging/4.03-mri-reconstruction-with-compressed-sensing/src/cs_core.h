// ===========================================================================
// src/cs_core.h  --  The shared __host__ __device__ CS-MRI "physics" core
// ---------------------------------------------------------------------------
// Project 4.3 : MRI Reconstruction with Compressed Sensing
//
// WHY THIS FILE EXISTS (PATTERNS.md section 2, the single most useful idiom)
//   Compressed-sensing MRI reconstruction is an ITERATIVE solver: the same handful
//   of tiny per-pixel operations (complex arithmetic, the soft-threshold proximal
//   operator, a gradient step) run thousands of times. If the CPU reference and the
//   GPU kernels used SEPARATE copies of those formulas, tiny differences would
//   accumulate over the iterations and the GPU-vs-CPU check would be meaningless.
//
//   So we put every per-element formula ONCE, here, as `__host__ __device__` inline
//   functions. reference_cpu.cpp includes this file through the plain host compiler;
//   kernels.cu includes it through nvcc. Both therefore execute BYTE-FOR-BYTE the
//   same arithmetic per element -- verification becomes a matter of the FFT library
//   (cuFFT vs our hand FFT) rather than of the iteration math.
//
//   HARD RULE: keep this header free of CUDA-only *types* and of `__global__`
//   kernels, so the host compiler can include it. Only the HD decorator macro and
//   plain inline functions live here.
//
// READ THIS BEFORE: reference_cpu.h, kernels.cuh. The science/math behind each
// formula is in ../THEORY.md.
// ===========================================================================
#pragma once

#include <cmath>     // std::sqrt, std::fabs, std::copysign  (host side)

// ---------------------------------------------------------------------------
// The HD decorator idiom.
//   When compiled by nvcc, __CUDACC__ is defined and we tag every shared function
//   `__host__ __device__` so it can run on BOTH the CPU and inside a kernel. When
//   compiled by the plain host compiler (cl.exe / g++), those keywords do not
//   exist, so CS_HD expands to nothing.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define CS_HD __host__ __device__
#else
#define CS_HD
#endif

// ---------------------------------------------------------------------------
// A minimal single-precision complex number.
//   MRI data lives in the complex Fourier domain: a k-space sample and an image
//   pixel are both complex. We deliberately use our OWN tiny struct (not
//   std::complex, not cuComplex) so the SAME type name works on host and device
//   and so the memory layout is trivially {float re; float im;} -- which is
//   bit-compatible with cuFFT's cufftComplex (float2) for a plain reinterpret.
//   Float (FP32) is what real scanners and cuFFT use; see THEORY "numerics".
// ---------------------------------------------------------------------------
struct Cplx {
    float re;   // real part
    float im;   // imaginary part
};

// c_make: build a complex from its parts (a readable constructor usable on device).
CS_HD inline Cplx c_make(float re, float im) { Cplx z; z.re = re; z.im = im; return z; }

// c_add / c_sub: componentwise complex addition / subtraction.
CS_HD inline Cplx c_add(Cplx a, Cplx b) { return c_make(a.re + b.re, a.im + b.im); }
CS_HD inline Cplx c_sub(Cplx a, Cplx b) { return c_make(a.re - b.re, a.im - b.im); }

// c_scale: multiply a complex by a real scalar (used by the gradient step size).
CS_HD inline Cplx c_scale(Cplx a, float s) { return c_make(a.re * s, a.im * s); }

// c_abs: magnitude |z| = sqrt(re^2 + im^2). This is the pixel BRIGHTNESS in the
// final magnitude image a radiologist would view (phase is usually discarded).
// We use sqrtf (the C float sqrt) rather than std::sqrt: sqrtf has both a host and
// a device implementation, so this __host__ __device__ function compiles cleanly
// for both targets and gives bit-identical results.
CS_HD inline float c_abs(Cplx z) { return sqrtf(z.re * z.re + z.im * z.im); }

// ---------------------------------------------------------------------------
// soft_threshold_real: the scalar proximal operator of the L1 norm.
//   prox_{lambda||.||_1}(x) = sign(x) * max(|x| - lambda, 0).
//   This is the mathematical heart of compressed sensing: it SHRINKS every
//   coefficient toward zero by lambda and sets small ones exactly to zero, which
//   is what enforces sparsity. Applied per real coefficient of the sparsifying
//   transform (here, per real/imag component -- see THEORY "the algorithm").
//   Branch-free formulation so host and device produce identical bits.
// ---------------------------------------------------------------------------
CS_HD inline float soft_threshold_real(float x, float lambda) {
    // |x| without calling std::fabs: on device std::fabs(float) would resolve to a
    // __host__ overload and warn, so we branch on the sign explicitly. This form is
    // also perfectly deterministic and identical on host and device.
    const float ax  = x >= 0.0f ? x : -x;             // |x|
    const float mag = ax - lambda;                    // shrink the magnitude by lambda
    const float pos = mag > 0.0f ? mag : 0.0f;        // clamp negatives to 0 (sparsify)
    // Restore the original sign of x. Branch-free sign multiply keeps host==device.
    const float sgn = x >= 0.0f ? 1.0f : -1.0f;
    return sgn * pos;
}

// soft_threshold_cplx: apply the L1 prox independently to the real and imaginary
// parts of a complex coefficient. Treating re and im separately is the standard
// "complex soft-threshold by components" used in CS-MRI toolboxes for a real-valued
// sparsity penalty; THEORY "real world" notes the magnitude-shrinkage variant.
CS_HD inline Cplx soft_threshold_cplx(Cplx z, float lambda) {
    return c_make(soft_threshold_real(z.re, lambda),
                  soft_threshold_real(z.im, lambda));
}

// ---------------------------------------------------------------------------
// data_consistency_residual: the per-sample forward-model residual in k-space.
//   The forward operator for single-coil Cartesian MRI is E = M . F, i.e. take the
//   2D FFT of the current image estimate, then keep only the SAMPLED k-space
//   positions (mask M). This function computes, for ONE k-space location:
//       r = mask ? (F{x}[k] - y[k]) : 0
//   where y is the measured (under-sampled) k-space. r is what the gradient step
//   pushes back through F^{-1}. Kept here so CPU and GPU compute it identically.
//     * fx    : the FFT of the current image at this k-space index
//     * y     : the measured k-space at this index (0 where unsampled)
//     * masked: 1 if this k-space position was acquired, else 0
// ---------------------------------------------------------------------------
CS_HD inline Cplx data_consistency_residual(Cplx fx, Cplx y, int masked) {
    if (masked) return c_sub(fx, y);   // sampled: penalize disagreement with data
    return c_make(0.0f, 0.0f);         // unsampled: no constraint (r = 0)
}
