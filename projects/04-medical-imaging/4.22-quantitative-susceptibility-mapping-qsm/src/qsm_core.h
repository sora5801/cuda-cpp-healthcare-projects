// ===========================================================================
// src/qsm_core.h  --  The shared, per-k-space-bin QSM math (CPU == GPU)
// ---------------------------------------------------------------------------
// Project 4.22 : Quantitative Susceptibility Mapping (QSM)
//
// WHY THIS FILE EXISTS  (PATTERNS.md section 2 -- the HD-macro idiom)
//   Every step of QSM dipole inversion that touches ONE k-space frequency bin
//   is a tiny, self-contained formula:
//       (a) the DIPOLE KERNEL value  D(k) = 1/3 - kz^2 / |k|^2         (forward)
//       (b) the TKD reciprocal       1 / D(k), but thresholded near D=0 (inverse)
//       (c) one Tikhonov gradient step in k-space (iterative inverse)
//   The 3-D Fourier transforms that BRACKET these steps differ in *how* they are
//   computed -- the CPU reference does a plain O(N^2) discrete Fourier transform
//   (DFT); the GPU does the identical transform with cuFFT in O(N log N). But the
//   per-bin arithmetic (a)-(c) MUST be byte-for-byte identical on both sides, or
//   the two reconstructions would drift apart for reasons that have nothing to do
//   with the FFT library. So we put (a)-(c) in ONE header as
//   `__host__ __device__` inline functions.  The CPU reference loops over them;
//   the GPU kernels call the same functions from one thread per bin. Same math,
//   both sides -> exact verification (down to double-precision round-off).
//
//   This header is included by BOTH:
//       * reference_cpu.cpp  (compiled by the host C++ compiler, cl.exe)
//       * kernels.cu / main.cu (compiled by nvcc)
//   so it MUST stay free of CUDA-only constructs (no __global__, no cufft types,
//   no <cuda_runtime.h>).  Only the HD decorators, guarded by __CUDACC__, plus a
//   pure-C++ Complex struct that both compilers understand.
//
// READ THIS BEFORE: reference_cpu.h, kernels.cuh.
// ===========================================================================
#pragma once

#include <cmath>     // std::sqrt, std::fabs

// ---------------------------------------------------------------------------
// The HD macro. Under nvcc (__CUDACC__ defined) a function tagged QSM_HD is
// compiled for BOTH the host and the device. Under the plain host compiler the
// decorators do not exist, so QSM_HD expands to nothing and the function is an
// ordinary inline host function. This is the single trick that buys CPU/GPU
// numerical parity (PATTERNS.md section 2).
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define QSM_HD __host__ __device__
#else
#define QSM_HD
#endif

// ---------------------------------------------------------------------------
// A minimal complex-number type shared by the CPU reference (which does its own
// hand-rolled DFT) and the GPU code (which converts to/from cuFFT's double2).
// We deliberately do NOT use std::complex or cufftDoubleComplex here so this
// header stays compilable by BOTH toolchains with identical layout
// (two contiguous doubles: real then imaginary). .re == real, .im == imaginary.
// ---------------------------------------------------------------------------
struct Complex {
    double re;   // real part
    double im;   // imaginary part
};

// Basic complex arithmetic (host+device). Kept explicit and commented so the
// learner can see exactly what each DFT/inversion step does.
QSM_HD inline Complex cplx(double re, double im) { Complex c; c.re = re; c.im = im; return c; }
QSM_HD inline Complex cadd(Complex a, Complex b) { return cplx(a.re + b.re, a.im + b.im); }
QSM_HD inline Complex cscale(Complex a, double s) { return cplx(a.re * s, a.im * s); }
// Complex multiply: (a+bi)(c+di) = (ac - bd) + (ad + bc)i.
QSM_HD inline Complex cmul(Complex a, Complex b) {
    return cplx(a.re * b.re - a.im * b.im,
                a.re * b.im + a.im * b.re);
}

// ---------------------------------------------------------------------------
// dipole_kernel: the value of the QSM dipole kernel D(k) at one frequency bin.
//
//   D(k) = 1/3 - (k . B0_hat)^2 / |k|^2
//
// With the main magnetic field B0 along +z (the standard convention), the unit
// field direction is (0,0,1), so (k . B0_hat) = kz and
//
//   D(k) = 1/3 - kz^2 / (kx^2 + ky^2 + kz^2).
//
// WHAT IT MEANS PHYSICALLY
//   A point magnetic dipole (a tiny blob of susceptibility) produces a
//   characteristic "bowtie" field pattern around it. The measured tissue field
//   perturbation is the true susceptibility distribution CONVOLVED with this
//   dipole response. The convolution theorem says convolution in space is a
//   pointwise MULTIPLY in k-space by D(k) -- so the forward model is simply
//   Fhat_field[k] = D(k) * Fhat_chi[k]. This function returns that D(k).
//
// THE ILL-POSED PART (the "magic angle" cone)
//   D(k) = 0 wherever kz^2 = |k|^2 / 3, i.e. on a double cone at ~54.7 degrees
//   (the magic angle) to B0. On that cone the forward model destroys all
//   information about chi, so the inverse (dividing by D) blows up. Every QSM
//   method is really a strategy for handling D(k) ~ 0 -- see tkd_reciprocal()
//   and the iterative Tikhonov path.
//
//   kx,ky,kz : the (signed) spatial-frequency coordinates of this bin. We pass
//              them already scaled so that |k| is dimensionless; the DC bin
//              (kx=ky=kz=0) has |k|^2 = 0 and we define D(0) = 0 there (no field
//              from a uniform susceptibility offset -- it is unobservable), which
//              also keeps the division guarded.
// ---------------------------------------------------------------------------
QSM_HD inline double dipole_kernel(double kx, double ky, double kz) {
    const double k2 = kx * kx + ky * ky + kz * kz;   // |k|^2
    if (k2 == 0.0) return 0.0;                        // DC: unobservable offset
    return (1.0 / 3.0) - (kz * kz) / k2;
}

// ---------------------------------------------------------------------------
// tkd_reciprocal: the Threshold-based K-space Division (TKD) inverse weight.
//
//   TKD reconstructs chi by DIVIDING the field's spectrum by D(k):
//       Fhat_chi[k] = Fhat_field[k] / D(k)
//   but near the magic-angle cone D(k) ~ 0 and 1/D(k) explodes, amplifying noise
//   into streaking artifacts. TKD's fix (Shmueli et al. 2009, Wharton 2010) is to
//   CLAMP the magnitude of D away from zero before inverting:
//
//       D_thr = sign(D) * max(|D|, thr)          (thr ~ 0.1..0.2, a few tenths)
//       weight = 1 / D_thr
//
//   Where |D| is already large the weight is the true 1/D (faithful inversion);
//   where |D| is tiny the weight is capped at +/- 1/thr (bounded, no blow-up).
//   This is the single most important trick in direct QSM -- a bias/variance
//   trade: a larger thr suppresses streaking but underestimates susceptibility.
//
//   D   : the dipole-kernel value at this bin (from dipole_kernel()).
//   thr : the threshold (0 < thr <= 1/3). Returns 0 at the DC bin (D==0) so the
//         mean susceptibility is set to zero (QSM is defined up to a constant).
// ---------------------------------------------------------------------------
QSM_HD inline double tkd_reciprocal(double D, double thr) {
    const double a = (D < 0.0) ? -D : D;             // |D|
    if (a < 1.0e-300) return 0.0;                     // exactly the DC/zero bin
    const double s = (D < 0.0) ? -1.0 : 1.0;          // sign(D)
    const double d_thr = s * (a > thr ? a : thr);     // clamp magnitude to >= thr
    return 1.0 / d_thr;                               // bounded inverse weight
}

// ---------------------------------------------------------------------------
// tikhonov_grad_step: one gradient-descent update of the iterative inverse, done
// entirely in k-space on ONE frequency bin.
//
// THE PROBLEM WE MINIMIZE
//   Iterative QSM solves a regularized least-squares problem
//       min_chi  || D .* Fchi - Ffield ||^2  +  alpha * || Fchi ||^2
//   where D.* is pointwise multiply by the dipole kernel in k-space. Because D is
//   diagonal in k-space, the whole thing DECOUPLES bin by bin: each frequency bin
//   is an independent 1-variable least-squares problem. The gradient w.r.t. the
//   (complex) unknown Fchi[k] is
//       g = 2*( D * (D*Fchi - Ffield) + alpha * Fchi )
//   and a gradient-descent step with step size `step` is
//       Fchi <- Fchi - step * g.
//
// WHY ITERATE AT ALL IF IT DECOUPLES?
//   For THIS simple Tikhonov model it does have a closed form (see the exact
//   Wiener weight below), and the CPU reference uses that closed form to make the
//   result exactly reproducible. The GRADIENT step is what generalizes to the
//   real MEDI-style problem where the regularizer (total variation, an edge mask)
//   couples neighbouring voxels and NO closed form exists -- then you truly need
//   O(100) iterations of "3-D FFT + this kind of gradient update", which is the
//   catalog's stated GPU bottleneck. We expose the gradient step so the kernel
//   and THEORY.md can teach that iterative structure honestly.
//
//   Fchi   : current estimate of chi's spectrum at this bin (complex)
//   Ffield : the measured field spectrum at this bin (complex, fixed)
//   D      : dipole-kernel value at this bin
//   alpha  : Tikhonov regularization weight (>= 0)
//   step   : gradient-descent step size
// ---------------------------------------------------------------------------
QSM_HD inline Complex tikhonov_grad_step(Complex Fchi, Complex Ffield,
                                         double D, double alpha, double step) {
    // residual r = D*Fchi - Ffield  (the forward model minus the data)
    const Complex r = cplx(D * Fchi.re - Ffield.re, D * Fchi.im - Ffield.im);
    // gradient g = 2*( D*r + alpha*Fchi )
    const Complex g = cplx(2.0 * (D * r.re + alpha * Fchi.re),
                           2.0 * (D * r.im + alpha * Fchi.im));
    // descent: Fchi <- Fchi - step * g
    return cplx(Fchi.re - step * g.re, Fchi.im - step * g.im);
}

// ---------------------------------------------------------------------------
// tikhonov_exact_weight: the closed-form minimizer of the Tikhonov problem above
// for one bin -- a WIENER-like filter. Setting the gradient to zero:
//       D*(D*Fchi - Ffield) + alpha*Fchi = 0
//   ->  (D^2 + alpha) * Fchi = D * Ffield
//   ->  Fchi = [ D / (D^2 + alpha) ] * Ffield
// so the exact per-bin inverse weight is  D / (D^2 + alpha).  As alpha -> 0 this
// tends to 1/D (unregularized, ill-posed); a positive alpha bounds it near the
// magic cone (there D~0 so the weight ~ D/alpha -> 0, i.e. those unreliable bins
// are gently zeroed). The CPU reference and the closed-form GPU path both use
// this so their results match to round-off, and the iterative gradient path is
// verified to CONVERGE toward it.
//
//   D     : dipole-kernel value at this bin
//   alpha : Tikhonov weight (> 0 keeps the denominator away from 0)
// ---------------------------------------------------------------------------
QSM_HD inline double tikhonov_exact_weight(double D, double alpha) {
    const double denom = D * D + alpha;
    if (denom < 1.0e-300) return 0.0;   // only when D==0 and alpha==0
    return D / denom;
}
