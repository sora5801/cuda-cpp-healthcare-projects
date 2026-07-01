// ===========================================================================
// src/pa_core.h  --  The SHARED per-pixel photoacoustic physics (CPU == GPU)
// ---------------------------------------------------------------------------
// Project 4.13 : Photoacoustic Image Reconstruction
//
// WHY THIS FILE EXISTS  (PATTERNS.md §2 -- the "__host__ __device__ core" idiom)
//   The single most valuable trick in this repo: put the per-element math in ONE
//   header, marked `__host__ __device__`, and include it from BOTH
//     * reference_cpu.cpp  (compiled by the plain host C++ compiler), and
//     * kernels.cu         (compiled by nvcc for the GPU),
//   so the CPU reference and the GPU kernel run BYTE-FOR-BYTE IDENTICAL math.
//   Verification then becomes *exact* (max error 0) instead of "close enough",
//   which is a far stronger correctness signal and a real teaching point.
//
//   To make that work this header must stay free of CUDA-only constructs
//   (no `__global__`, no `<<< >>>`, no device-only types) so the host compiler
//   can also digest it. It contains only: the problem struct, and a couple of
//   small `PA_HD inline` helper functions.
//
// THE ONE FORMULA THIS FILE OWNS: delay-and-sum (DAS) backprojection.
//   A photoacoustic (PA) source emits a pressure wave that travels outward at
//   the speed of sound `c`. A point sensor at position `p_s` records a pressure
//   time-series `g_s(t)`. To ask "how much source was at image point `x`?", we
//   look up, in every sensor's trace, the sample that would have ARRIVED FROM
//   `x`: it left at emission time 0 and reaches sensor `s` after the travel time
//         tau_s(x) = |x - p_s| / c.
//   Reading g_s at t = tau_s(x) and SUMMING over all sensors constructively
//   reinforces the true source location and averages away everything else.
//   That sum, times a normalization, is our reconstructed image value b(x).
//   (This is the discrete "universal back-projection" / DAS estimator; see
//   THEORY.md for the exact continuous formula it approximates.)
//
// READ THIS BEFORE: reference_cpu.h, kernels.cuh.
// ===========================================================================
#pragma once

#include <cstddef>   // std::size_t
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// PA_HD: expands to `__host__ __device__` when compiled by nvcc (so the helper
// below can run on the GPU), and to NOTHING when compiled by the host compiler
// (which does not understand those keywords). This is the portable-inline idiom
// from PATTERNS.md §2 -- the reason CPU and GPU produce identical numbers.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define PA_HD __host__ __device__
#else
#define PA_HD
#endif

// ---------------------------------------------------------------------------
// PAProblem: everything needed to reconstruct one 2-D photoacoustic image.
//
//   Geometry / acquisition:
//     n_sensors  number of point ultrasound sensors around the object.
//     n_samples  time samples recorded per sensor (fast-time axis).
//     dt         sampling interval in SECONDS (e.g. 2.5e-8 s = 40 MHz ADC).
//     c          speed of sound in METRES/SECOND (~1500 m/s in soft tissue).
//
//   Reconstruction grid:
//     img        output image side length in pixels (image is img x img).
//     world_half the image covers the square [-world_half, +world_half]^2 in
//                METRES; pixel spacing is 2*world_half/(img-1).
//
//   Data (flat, row-major):
//     sx[s], sy[s]                    sensor s position in metres.
//     sig[s*n_samples + t]            pressure recorded by sensor s at sample t
//                                     (arbitrary pressure units; see data/README).
//
// All lengths are in metres and all times in seconds, so distance/c has units of
// seconds and divides cleanly by dt to give a (fractional) sample index.
// ---------------------------------------------------------------------------
struct PAProblem {
    int   n_sensors = 0;      // number of sensors on the measurement aperture
    int   n_samples = 0;      // time samples per sensor trace
    int   img       = 0;      // reconstructed image side (pixels)
    float dt        = 0.0f;   // sample period [s]
    float c         = 0.0f;   // speed of sound [m/s]
    float world_half = 0.0f;  // image half-extent [m]; image spans [-W,W]^2
    std::vector<float> sx;    // [n_sensors] sensor x [m]
    std::vector<float> sy;    // [n_sensors] sensor y [m]
    std::vector<float> sig;   // [n_sensors * n_samples] pressure traces
};

// ---------------------------------------------------------------------------
// pa_sample_trace: read sensor `s`'s time-series at a FRACTIONAL sample index
// `fidx` using linear interpolation, returning 0 when the index falls outside
// the recorded window.
//
//   Why linear interpolation? The travel time tau = dist/c almost never lands
//   exactly on a sample boundary, so we blend the two neighbouring samples. This
//   is the same interpolation CT backprojection (4.01) does on the detector; on
//   a GPU it is what texture hardware performs for free (see THEORY.md §GPU).
//
//   Parameters:
//     sig        pointer to the start of this sensor's trace (n_samples floats).
//     n_samples  length of the trace (guards the ends).
//     fidx       fractional sample index = tau / dt.
//   Returns the interpolated pressure, or 0 outside [0, n_samples-1].
//
//   Marked PA_HD so the SAME code runs in reference_cpu.cpp and in the kernel.
// ---------------------------------------------------------------------------
PA_HD inline float pa_sample_trace(const float* sig, int n_samples, float fidx) {
    // Reject arrivals that fall before the first / after the last recorded
    // sample: those carry no measured information, so they contribute nothing.
    if (fidx < 0.0f || fidx > (float)(n_samples - 1)) return 0.0f;
    // Integer part = lower neighbour; fractional part = interpolation weight.
    const int   i0 = (int)fidx;               // floor for non-negative fidx
    const float w  = fidx - (float)i0;        // in [0,1): distance to i0
    // Guard the exact-last-sample case (i0 == n_samples-1, w == 0): i0+1 would
    // read past the end, but w==0 means the second term is zeroed anyway; we
    // still must not dereference sig[i0+1], so branch on it.
    const float a = sig[i0];
    const float b = (i0 + 1 < n_samples) ? sig[i0 + 1] : a;
    return a * (1.0f - w) + b * w;            // linear blend of the two neighbours
}

// ---------------------------------------------------------------------------
// pa_pixel_das: the CORE reconstruction formula for a single output pixel.
// Computes the delay-and-sum backprojection value at world point (wx, wy).
//
//   For every sensor s:
//     dist   = sqrt((wx - sx)^2 + (wy - sy)^2)   Euclidean distance to sensor
//     tau    = dist / c                          acoustic travel time [s]
//     fidx   = tau / dt                          fractional sample index
//     value += g_s(fidx)                         interpolated recorded pressure
//   Then multiply by (1 / n_sensors) so the estimate is the MEAN sensor
//   response, independent of how many sensors we happen to have.
//
//   This is exactly the loop the CPU reference runs serially and the GPU kernel
//   runs one-thread-per-pixel. Keeping it here guarantees they agree exactly.
//
//   Parameters (all SI units):
//     wx, wy      world coordinates of the pixel [m].
//     sx, sy      [n_sensors] sensor coordinates [m].
//     sig         [n_sensors * n_samples] pressure traces (row-major per sensor).
//     n_sensors   number of sensors to sum over.
//     n_samples   samples per trace.
//     inv_c       precomputed 1/c [s/m] (avoids a divide in the inner loop).
//     inv_dt      precomputed 1/dt [1/s].
//     inv_ns      precomputed 1/n_sensors normalization.
//   Returns the reconstructed image value at (wx, wy).
// ---------------------------------------------------------------------------
PA_HD inline float pa_pixel_das(float wx, float wy,
                                const float* sx, const float* sy,
                                const float* sig, int n_sensors, int n_samples,
                                float inv_c, float inv_dt, float inv_ns) {
    float acc = 0.0f;   // running delay-and-sum accumulator for this pixel
    for (int s = 0; s < n_sensors; ++s) {
        const float dx = wx - sx[s];               // vector from sensor to pixel
        const float dy = wy - sy[s];
        const float dist = sqrtf(dx * dx + dy * dy);  // |x - p_s| [m]
        const float fidx = dist * inv_c * inv_dt;     // (dist/c)/dt = sample index
        // Point this sensor's trace base pointer and interpolate at fidx.
        acc += pa_sample_trace(sig + (size_t)s * n_samples, n_samples, fidx);
    }
    return acc * inv_ns;   // mean over sensors -> the reconstructed pixel value
}
