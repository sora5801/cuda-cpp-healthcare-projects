// ===========================================================================
// src/ctf_model.h  --  Shared (host + device) cryo-EM CTF physics
// ---------------------------------------------------------------------------
// Project 2.11 : Cryo-EM CTF Estimation & Particle Picking
//
// WHY THIS FILE EXISTS (the single most important idiom in this repo)
//   The per-element PHYSICS of the contrast transfer function lives here, in ONE
//   header, as `__host__ __device__` inline functions. The CPU reference
//   (reference_cpu.cpp, compiled by cl.exe) AND the GPU kernels (kernels.cu,
//   compiled by nvcc) both #include this file and call the SAME functions. That
//   means the model CTF the GPU grid-search fits is byte-for-byte identical to
//   the one the CPU reference fits -- so "GPU == CPU" verification can be EXACT
//   on the integer argmax (which defocus wins) and tight (1e-5) on the scores.
//   See PATTERNS.md §2 and THEORY.md "How we verify correctness".
//
//   CTF_HD expands to `__host__ __device__` under nvcc and to nothing under the
//   host compiler (which has never heard of those decorators). Keep this header
//   free of CUDA-only types (no float2, no __global__) so cl.exe can include it.
//
// THE SCIENCE IN ONE PARAGRAPH
//   A cryo-EM microscope does not image the specimen faithfully: lens defocus and
//   spherical aberration impose an oscillating, sign-flipping transfer function on
//   spatial frequencies. For a weak-phase object the recorded image's Fourier
//   amplitude at spatial frequency k (1/Angstrom) is multiplied by
//       CTF(k) = sin( chi(k) )                                            (1)
//   where the phase aberration (no astigmatism, the teaching case) is
//       chi(k) = pi*lambda*dz*k^2  -  (pi/2)*Cs*lambda^3*k^4  +  phi      (2)
//   with electron wavelength lambda (A), defocus dz (A, under-focus > 0),
//   spherical aberration Cs (A), and an extra phase shift phi (rad, e.g. a phase
//   plate or the constant amplitude-contrast term). Because CTF oscillates, its
//   square |CTF(k)|^2 -- which is what survives in a POWER spectrum -- shows
//   bright concentric "Thon rings" whose spacing encodes dz. Estimating the CTF
//   is therefore: measure the radial power-spectrum profile, then find the dz
//   whose |CTF|^2 ring pattern best matches it. That is exactly what we compute.
//
// READ THIS AFTER: reference_cpu.h (which owns the Micrograph / CtfParams types).
// READ THIS BEFORE: kernels.cu (the GPU twin of the CPU fitter).
// ===========================================================================
#pragma once

#include <cmath>     // std::sin, std::sqrt (host + device both have these)
#include <cstddef>   // std::size_t

// CTF_HD: the host/device portability shim (PATTERNS.md §2, "HD-macro idiom").
#ifdef __CUDACC__
#define CTF_HD __host__ __device__
#else
#define CTF_HD
#endif

#ifndef CTF_PI
#define CTF_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// CtfParams: the fixed microscope/optics constants for a session. Only `dz`
// (defocus) is being SEARCHED; everything else is known from the instrument.
// Units are spelled out because mixing Angstroms and metres is the classic CTF
// bug. All lengths are in Angstrom (A); angles in radians.
// ---------------------------------------------------------------------------
struct CtfParams {
    double lambda;      // electron wavelength (A), e.g. 0.0197 A at 300 kV
    double cs;          // spherical aberration (A), e.g. 2.7e7 A == 2.7 mm
    double amp_contrast;// amplitude-contrast fraction in [0,1], e.g. 0.10
    double pixel_size;  // detector pixel size on the specimen (A/pixel)
    int    n;           // micrograph side length in pixels (square, power of 2)
};

// ---------------------------------------------------------------------------
// ctf_extra_phase: the constant phase term phi in eq.(2) contributed by the
// amplitude contrast. The convention (CTFFIND4, RELION) writes the sin() form
// with an added phase phi = atan( ac / sqrt(1-ac^2) ), i.e. asin(ac), so that
// at k=0 the transfer is -ac (a small flat contrast), not zero. We expose it as
// its own function so the CPU and GPU use the identical constant.
// ---------------------------------------------------------------------------
CTF_HD inline double ctf_extra_phase(double amp_contrast) {
    // asin(ac) is the phase whose sine equals ac; equivalent to the atan form.
    return std::asin(amp_contrast);
}

// ---------------------------------------------------------------------------
// ctf_value: the (signed) CTF of eq.(1)-(2) at spatial frequency k (1/A) for a
// given defocus dz (A). This is THE one true formula; both the forward model
// (synthetic data) and the fitter call it.
//   k    : spatial frequency magnitude (cycles per Angstrom), >= 0
//   dz   : defocus (A); under-focus is positive in this convention
//   p    : optics constants (lambda, cs, amp_contrast)
//   returns CTF in [-1, 1].
// ---------------------------------------------------------------------------
CTF_HD inline double ctf_value(double k, double dz, const CtfParams& p) {
    const double k2 = k * k;                    // k^2  (1/A^2)
    const double k4 = k2 * k2;                  // k^4  (1/A^4)
    // chi(k): the phase aberration of eq.(2). The defocus term is +, the Cs term
    // is - in this sign convention (Mindell & Grigorieff 2003 / CTFFIND).
    const double chi = CTF_PI * p.lambda * dz * k2
                     - 0.5 * CTF_PI * p.cs * p.lambda * p.lambda * p.lambda * k4
                     + ctf_extra_phase(p.amp_contrast);
    return std::sin(chi);                       // eq.(1)
}

// ---------------------------------------------------------------------------
// ctf_squared: |CTF(k)|^2. This is what appears in a POWER spectrum (the squared
// Fourier amplitude), so it is the quantity we match against the observed radial
// profile. Always in [0,1]; its zeros are the dark gaps between Thon rings.
// ---------------------------------------------------------------------------
CTF_HD inline double ctf_squared(double k, double dz, const CtfParams& p) {
    const double c = ctf_value(k, dz, p);
    return c * c;
}

// ---------------------------------------------------------------------------
// freq_of_bin: convert a radial bin index r (in pixels, 0..nbins-1) to the
// spatial frequency k (1/A) it represents. In an N x N FFT the Nyquist frequency
// 0.5/pixel_size sits at radius N/2 pixels, so k(r) = (r/(N/2)) * (0.5/dx) =
// r / (N * dx). We pass nyquist_k = 0.5/dx and half = N/2 to avoid recomputing.
//   r        : radial bin (pixels from the DC/centre)
//   half     : N/2 (the pixel radius of Nyquist)
//   nyquist_k: 0.5 / pixel_size (1/A) -- the highest representable frequency
// ---------------------------------------------------------------------------
CTF_HD inline double freq_of_bin(int r, int half, double nyquist_k) {
    return (static_cast<double>(r) / static_cast<double>(half)) * nyquist_k;
}

// ---------------------------------------------------------------------------
// ncc_model_vs_profile: the FIT SCORE. Given the observed radial power profile
// `prof[0..nbins-1]` and a candidate defocus dz, build the model |CTF(k)|^2 over
// the SAME bins and return the normalized cross-correlation (Pearson r) between
// model and observation over the fitting band [r_lo, r_hi). NCC is the standard
// CTF-fit objective (CTFFIND maximizes a closely-related cross-correlation): it
// is invariant to the model's and data's overall scale/offset, so we do not have
// to match the (unknown) envelope amplitude -- only the RING POSITIONS, which is
// what carries the defocus. Higher is better; the best dz maximizes this.
//
//   prof      : observed radial profile (length nbins), already background-flattened
//   nbins     : number of radial bins
//   r_lo,r_hi : inclusive-exclusive bin band used for fitting (skip the DC spike
//               near r=0 and the noisy highest frequencies)
//   dz        : candidate defocus (A)
//   p         : optics constants
//   half      : N/2
//   nyquist_k : 0.5 / pixel_size
//
// Implementation note: this is a single pass computing the means then a second
// pass for the covariance/variances -- O(nbins). It is deliberately written with
// plain doubles and no atomics so the CPU loop and the one-thread-per-candidate
// GPU version produce identical bits.
// ---------------------------------------------------------------------------
CTF_HD inline double ncc_model_vs_profile(const double* prof, int nbins,
                                          int r_lo, int r_hi, double dz,
                                          const CtfParams& p,
                                          int half, double nyquist_k) {
    const int m = r_hi - r_lo;                  // number of bins in the fit band
    if (m <= 1) return -2.0;                     // degenerate -> impossible score

    // Pass 1: accumulate sums to form the two means (model and data).
    double sum_m = 0.0, sum_d = 0.0;
    for (int r = r_lo; r < r_hi; ++r) {
        const double k = freq_of_bin(r, half, nyquist_k);
        sum_m += ctf_squared(k, dz, p);
        sum_d += prof[r];
    }
    const double mean_m = sum_m / m;
    const double mean_d = sum_d / m;

    // Pass 2: covariance and the two variances (Pearson numerator/denominator).
    double cov = 0.0, var_m = 0.0, var_d = 0.0;
    for (int r = r_lo; r < r_hi; ++r) {
        const double k  = freq_of_bin(r, half, nyquist_k);
        const double dm = ctf_squared(k, dz, p) - mean_m;   // model deviation
        const double dd = prof[r] - mean_d;                  // data deviation
        cov   += dm * dd;
        var_m += dm * dm;
        var_d += dd * dd;
    }
    const double denom = std::sqrt(var_m * var_d);
    if (denom <= 0.0) return -2.0;               // a flat model/data -> undefined
    return cov / denom;                          // Pearson r in [-1, 1]
}
