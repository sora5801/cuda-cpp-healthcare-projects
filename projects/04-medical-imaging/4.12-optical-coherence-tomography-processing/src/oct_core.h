// ===========================================================================
// src/oct_core.h  --  The ONE TRUE per-sample OCT math (CPU + GPU share this)
// ---------------------------------------------------------------------------
// Project 4.12 : Optical Coherence Tomography Processing (SD-OCT reconstruction)
//
// WHY THIS HEADER EXISTS (PATTERNS.md #2 -- the __host__ __device__ core)
//   Spectral-domain OCT reconstruction is a per-spectral-sample transform of the
//   raw interferogram (remove DC, apply a window, apply a dispersion-correction
//   phase) followed by a length-N FFT per A-scan. The GPU path (kernels.cu) and
//   the CPU reference (reference_cpu.cpp) MUST perform byte-for-byte-identical
//   per-sample math, otherwise "GPU == CPU" verification degrades from an exact
//   check into a fuzzy one. The trick is to put that per-sample math in ONE place,
//   as `__host__ __device__` inline functions, and call it from both sides:
//
//       * reference_cpu.cpp  -- compiled by the HOST compiler (cl.exe / g++)
//       * kernels.cu         -- compiled by nvcc for the DEVICE
//
//   Under nvcc the OCT_HD macro expands to `__host__ __device__` so the function
//   compiles for both targets; under the plain host compiler it expands to
//   nothing (those decorators do not exist there). Either way the SAME source
//   text -- the same additions, multiplications, sin/cos -- is what runs, so the
//   two paths agree to the last representable bit of the shared operations.
//
//   HARD RULE: keep this header free of CUDA-only constructs (no `__global__`, no
//   `float2`/`cufftComplex`, no <cuda_runtime.h>). It is included by a pure C++
//   translation unit, so it must compile as plain C++17. We therefore carry the
//   tiny complex helper as a plain struct `Cplx` and do complex arithmetic by
//   hand -- transparent, and identical on both sides.
//
// READ THIS BEFORE: reference_cpu.h (data model), kernels.cuh (GPU wrappers).
// ===========================================================================
#pragma once

#include <cmath>     // std::sin, std::cos, std::sqrt, std::log10, std::fabs
#include <cstddef>   // std::size_t

// ---------------------------------------------------------------------------
// OCT_HD: the host/device decoration macro (PATTERNS.md #2).
//   * Compiled by nvcc  (__CUDACC__ is defined) -> `__host__ __device__`, so the
//     inline function is emitted for BOTH the CPU and the GPU.
//   * Compiled by cl.exe/g++ (host only)         -> empty, because those keywords
//     are CUDA-specific and would be a syntax error to a plain C++ compiler.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define OCT_HD __host__ __device__
#else
#define OCT_HD
#endif

// ---------------------------------------------------------------------------
// Cplx: a minimal complex number used by BOTH sides.
//   We deliberately do NOT use std::complex (host) or cufftComplex/float2
//   (device) in the shared core, because those are different types on the two
//   compilers and would break the "one true math" guarantee. A plain POD struct
//   of two doubles, with hand-written add/mul, is identical everywhere and makes
//   the arithmetic explicit for the learner. Double precision here keeps the CPU
//   reference exact; kernels.cu mirrors these ops in double as well (see THEORY
//   "Numerical considerations").
// ---------------------------------------------------------------------------
struct Cplx {
    double re;   // real part
    double im;   // imaginary part
};

// Complex construction / addition / multiplication -- spelled out on purpose.
OCT_HD inline Cplx cplx(double re, double im) { Cplx z; z.re = re; z.im = im; return z; }

OCT_HD inline Cplx cadd(Cplx a, Cplx b) { return cplx(a.re + b.re, a.im + b.im); }

// (a.re + i a.im) * (b.re + i b.im) = (a.re b.re - a.im b.im) + i(a.re b.im + a.im b.re)
OCT_HD inline Cplx cmul(Cplx a, Cplx b) {
    return cplx(a.re * b.re - a.im * b.im,
                a.re * b.im + a.im * b.re);
}

// |z|^2 -- squared magnitude, cheaper than a magnitude (no sqrt) and all we need
// for the intensity image before the log.
OCT_HD inline double cabs2(Cplx z) { return z.re * z.re + z.im * z.im; }

// ---------------------------------------------------------------------------
// hann_window(i, n): the i-th coefficient of a length-n Hann (raised-cosine)
//   window, w[i] = 0.5 * (1 - cos(2*pi*i/(n-1))), i = 0..n-1.
//
//   WHY WINDOW AT ALL? The raw OCT spectrum is a finite slice of an (in principle)
//   infinite interferogram. FFT-ing a hard-truncated slice convolves each depth
//   reflector with a sinc, whose tall side lobes smear signal across depths
//   (spectral leakage). Multiplying by a smooth taper that goes to zero at both
//   ends suppresses those side lobes, at the cost of a slightly wider main lobe
//   (axial resolution). This is a standard OCT preprocessing step.
//
//   n <= 1 is guarded to avoid a divide-by-zero on (n-1).
// ---------------------------------------------------------------------------
OCT_HD inline double hann_window(int i, int n) {
    if (n <= 1) return 1.0;
    const double PI = 3.14159265358979323846;
    return 0.5 * (1.0 - std::cos(2.0 * PI * static_cast<double>(i) / (n - 1)));
}

// ---------------------------------------------------------------------------
// dispersion_phase(i, n, a2, a3): the dispersion-correction phase applied to
//   spectral sample i of a length-n A-scan.
//
//   THE PHYSICS. In a real OCT interferometer the sample and reference arms have
//   slightly different amounts of dispersive material (glass, water, tissue). A
//   dispersive medium makes the optical phase a NON-LINEAR function of wavenumber
//   k, which in the spectral interferogram appears as a k-dependent phase error
//   phi(k) ~= a2*(k-k0)^2 + a3*(k-k0)^3 + ...  (2nd + 3rd order dominate). Left
//   uncorrected, this phase broadens every reflector's point-spread function --
//   depth peaks smear out and axial resolution collapses.
//
//   THE FIX ("numerical dispersion compensation"). Multiply the (complexified)
//   spectrum by exp(-i * phi(k)) BEFORE the FFT, cancelling the phase error so
//   each reflector collapses back to a sharp peak. Here we index k by the sample
//   index i, centred so k0 is the middle sample: kc = (i - (n-1)/2) / n maps the
//   band to roughly [-0.5, 0.5]. We return the phase; the caller forms
//   exp(-i*phase) = cos(phase) - i*sin(phase) and multiplies.
//
//   a2, a3 are the (unitless, in this normalised k) dispersion coefficients. In
//   the synthetic data we INJECT a known (a2,a3) into the raw spectra and then
//   remove it here -- so the demo visibly shows compensation sharpening the peaks
//   (see THEORY "How we verify"). Set a2=a3=0 to disable compensation.
// ---------------------------------------------------------------------------
OCT_HD inline double dispersion_phase(int i, int n, double a2, double a3) {
    const double kc = (static_cast<double>(i) - 0.5 * (n - 1)) / static_cast<double>(n);
    return a2 * kc * kc + a3 * kc * kc * kc;
}

// ---------------------------------------------------------------------------
// preprocess_sample(raw, dc, i, n, a2, a3): turn ONE raw real spectral sample
//   into the complex FFT input for that sample. This is the fused per-sample
//   pipeline shared by CPU and GPU -- the single source of truth:
//
//     1. background/DC removal : s = raw - dc      (dc = per-A-scan mean)
//     2. windowing             : s *= hann_window(i, n)
//     3. complexify + disperse : z = s * exp(-i * dispersion_phase(i,...))
//
//   The result feeds the length-n FFT (cufftExecC2C on the GPU; the naive DFT on
//   the CPU). Because every arithmetic op here is identical on both compilers,
//   the FFT inputs are identical, so the reconstructions match within the tiny
//   FFT-vs-DFT rounding tolerance documented in main.cu.
// ---------------------------------------------------------------------------
OCT_HD inline Cplx preprocess_sample(double raw, double dc, int i, int n,
                                     double a2, double a3) {
    const double s = (raw - dc) * hann_window(i, n);         // steps 1 + 2
    const double phi = dispersion_phase(i, n, a2, a3);        // step 3 phase
    // exp(-i*phi) = cos(phi) - i sin(phi); multiply the real, windowed sample by it.
    return cplx(s * std::cos(phi), -s * std::sin(phi));
}

// ---------------------------------------------------------------------------
// log_intensity(power, floor_db): convert a linear depth-profile power |A|^2
//   into the decibel scale that OCT B-scans are always displayed in:
//       I_dB = 10 * log10(power / power_max)     (clamped at floor_db)
//   Here we pass an already-normalised `power` (0..1 relative to the A-scan max)
//   so the brightest reflector sits at 0 dB and everything below floor_db is
//   clamped to floor_db (the usual display-range clip). Returns dB (<= 0).
// ---------------------------------------------------------------------------
OCT_HD inline double log_intensity(double power_norm, double floor_db) {
    if (power_norm <= 0.0) return floor_db;
    double db = 10.0 * std::log10(power_norm);
    return db < floor_db ? floor_db : db;
}
