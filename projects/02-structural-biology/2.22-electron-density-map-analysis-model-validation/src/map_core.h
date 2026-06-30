// ===========================================================================
// src/map_core.h  --  Shared __host__ __device__ core for map validation
// ---------------------------------------------------------------------------
// Project 2.22 : Electron Density Map Analysis & Model Validation
//
// THE HD-MACRO IDIOM (PATTERNS.md §2 -- "the single most useful idiom").
//   Both the CPU reference (reference_cpu.cpp, compiled by cl.exe/g++) and the
//   GPU kernels (kernels.cu, compiled by nvcc) must run BYTE-FOR-BYTE identical
//   per-voxel math, so that verification is exact instead of approximate. We
//   achieve that by putting the per-voxel physics in ONE header as inline
//   functions decorated `__host__ __device__` when compiled by nvcc, and as
//   plain inline functions when compiled by the host compiler.
//
//   RULES for this header (so the host compiler can include it):
//     * NO __global__ kernels here (those live in kernels.cuh/.cu).
//     * NO CUDA-only types here (float2/cufftComplex stay in kernels.cu).
//     * Only the small, self-contained scalar formulas shared by both paths.
//
// WHAT LIVES HERE
//   1. Cplx          -- a tiny POD complex number (so the header needs no CUDA).
//   2. shell_index() -- map a reciprocal-space voxel (kx,ky,kz) to its spherical
//                       resolution shell (the "ring averaging" of FSC).
//   3. fsc_terms()   -- the three real accumulands of one voxel's FSC term.
//   4. pearson_from_sums() -- close the real-space correlation coefficient (RSCC)
//                       from running sums (so CPU and GPU finish identically).
//
// READ THIS AFTER: reference_cpu.h (defines DensityMap).  Used BY: reference_cpu.cpp
//   and kernels.cu.  The science/derivation is in ../THEORY.md.
// ===========================================================================
#pragma once

#include <cmath>   // std::sqrt, std::floor (host); nvcc maps these to device intrinsics

// HD: expands to `__host__ __device__` under nvcc (so the SAME inline function
// is emitted for BOTH the CPU and the GPU), and to nothing for the host
// compiler (which has never heard of those decorators). This is the crux of the
// CPU/GPU parity trick -- see PATTERNS.md §2.
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ---------------------------------------------------------------------------
// Cplx: a minimal complex number as a plain struct of two doubles.
//   We deliberately do NOT use std::complex (its operator* is not guaranteed
//   __device__-callable) nor cufftComplex (CUDA-only, single precision) here.
//   The FFT output is copied into Cplx[] on the host before binning, so this
//   one type works on both sides of the verification.
//     .re = real part, .im = imaginary part.
// ---------------------------------------------------------------------------
struct Cplx {
    double re;
    double im;
};

// ---------------------------------------------------------------------------
// fft_freq: the signed integer frequency index of FFT bin `i` along an axis of
//   length `n`. NumPy's np.fft.fftfreq convention (in cycles-per-box, i.e. the
//   integer multiple of 1/n):
//       0, 1, 2, ..., n/2-1,  -n/2, ..., -2, -1     (for even n)
//   The first half are the positive frequencies; the second half wrap to the
//   negative frequencies. We need the SIGNED frequency so that the distance to
//   the origin in reciprocal space (|k|) is correct for the spherical shells.
// ---------------------------------------------------------------------------
HD inline int fft_freq(int i, int n) {
    // Bins [0 .. n/2] are non-negative; bins (n/2 .. n) are negative (i - n).
    return (i <= n / 2) ? i : (i - n);
}

// ---------------------------------------------------------------------------
// shell_radius: the Euclidean radius |k| (in integer-frequency units) of a
//   reciprocal-space voxel, given the signed frequency indices on each axis.
//   For a cubic box this is sqrt(kx^2 + ky^2 + kz^2); it is the spatial
//   frequency that FSC averages over (each spherical "shell" is one |k|).
// ---------------------------------------------------------------------------
HD inline double shell_radius(int kx, int ky, int kz) {
    const double r2 = static_cast<double>(kx) * kx
                    + static_cast<double>(ky) * ky
                    + static_cast<double>(kz) * kz;
    return std::sqrt(r2);
}

// ---------------------------------------------------------------------------
// shell_index: assign a reciprocal-space voxel to its FSC shell (an integer
//   bin of |k|). We bin by ROUNDING the radius to the nearest integer, the
//   standard FSC shell definition (shell s collects all voxels with
//   round(|k|) == s). Voxels at the box corners have |k| up to ~sqrt(3)*n/2;
//   the caller sizes the shell array to `max_shell(n)` to hold them all.
//
//   Why round(): it makes each shell ~1 frequency unit thick, which is the
//   reciprocal-space resolution increment of an n-voxel box. Returns the shell
//   index s >= 0.
// ---------------------------------------------------------------------------
HD inline int shell_index(int kx, int ky, int kz) {
    const double r = shell_radius(kx, ky, kz);
    // +0.5 then floor == round-half-up for non-negative r. (No std::round so
    // host and device agree exactly on the tie-break rule.)
    return static_cast<int>(std::floor(r + 0.5));
}

// ---------------------------------------------------------------------------
// max_shell: how many shells an n^3 box needs. The farthest voxel is the box
//   corner at frequency (n/2, n/2, n/2), radius (n/2)*sqrt(3); round() of that
//   is the largest shell index, so we allocate one more. A small, exact bound
//   shared by host and device so both index the same array length.
// ---------------------------------------------------------------------------
HD inline int max_shell(int n) {
    const double rmax = static_cast<double>(n / 2) * 1.7320508075688772;  // (n/2)*sqrt(3)
    return static_cast<int>(std::floor(rmax + 0.5)) + 1;
}

// ---------------------------------------------------------------------------
// fsc_accumulate: add ONE reciprocal-space voxel's contribution to the three
//   running sums of its shell. Fourier Shell Correlation of two maps with
//   Fourier transforms F1, F2 is, per shell s:
//
//                  Re( Σ_{k in s} F1(k) · conj(F2(k)) )
//       FSC(s) = ---------------------------------------------------
//                 sqrt( Σ |F1(k)|² · Σ |F2(k)|² )            (over k in s)
//
//   This function computes the three per-voxel accumulands:
//       cross += Re(F1 · conj(F2)) = re1*re2 + im1*im2
//       p1    += |F1|²             = re1² + im1²
//       p2    += |F2|²             = re2² + im2²
//   The CPU loops this over every voxel; the GPU calls it from one thread per
//   voxel. Identical formula on both sides => exact agreement (PATTERNS.md §2).
//   Pointers are in/out running sums for shell `s` (caller owns the arrays).
// ---------------------------------------------------------------------------
HD inline void fsc_accumulate(const Cplx& f1, const Cplx& f2,
                              double* cross, double* p1, double* p2) {
    *cross += f1.re * f2.re + f1.im * f2.im;   // Re(F1 · conj(F2))
    *p1    += f1.re * f1.re + f1.im * f1.im;    // |F1|²
    *p2    += f2.re * f2.re + f2.im * f2.im;    // |F2|²
}

// ---------------------------------------------------------------------------
// fsc_from_sums: close one shell's FSC value from its three accumulated sums.
//   Guards the empty/zero-power shell (returns 0) so a shell with no voxels or
//   a flat (zero-variance) map does not divide by zero.
// ---------------------------------------------------------------------------
HD inline double fsc_from_sums(double cross, double p1, double p2) {
    const double denom = std::sqrt(p1 * p2);
    return (denom > 0.0) ? (cross / denom) : 0.0;
}

// ---------------------------------------------------------------------------
// pearson_from_sums: finish the REAL-SPACE correlation coefficient (RSCC)
//   between two maps a,b from running sums collected over N voxels:
//       Sa = Σ a,  Sb = Σ b,  Saa = Σ a²,  Sbb = Σ b²,  Sab = Σ a·b
//   Pearson r = (N·Sab - Sa·Sb) / sqrt( (N·Saa - Sa²)(N·Sbb - Sb²) ).
//   RSCC is the standard "does the model fit the density" score in real space
//   (Phenix/CCP4 report it per residue). The same closing formula runs on CPU
//   and GPU, so the two RSCC values match to rounding.
// ---------------------------------------------------------------------------
HD inline double pearson_from_sums(double n, double Sa, double Sb,
                                   double Saa, double Sbb, double Sab) {
    const double cov   = n * Sab - Sa * Sb;
    const double var_a = n * Saa - Sa * Sa;
    const double var_b = n * Sbb - Sb * Sb;
    const double denom = std::sqrt(var_a * var_b);
    return (denom > 0.0) ? (cov / denom) : 0.0;
}
