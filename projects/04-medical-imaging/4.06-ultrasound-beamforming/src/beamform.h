// ===========================================================================
// src/beamform.h  --  The ONE TRUE per-pixel-per-element DAS physics (shared)
// ---------------------------------------------------------------------------
// Project 4.6 : Ultrasound Beamforming (Delay-and-Sum)
//
// WHY THIS HEADER EXISTS  (PATTERNS.md §2 -- the __host__ __device__ core idiom)
//   The single most useful trick for CPU/GPU parity: put the per-element
//   physics in ONE header as `__host__ __device__` inline functions, so the CPU
//   reference (reference_cpu.cpp, compiled by cl.exe/g++) and the GPU kernel
//   (kernels.cu, compiled by nvcc) run BYTE-FOR-BYTE-IDENTICAL math. Then the
//   GPU-vs-CPU verification in main.cu is a *tight* check, not a fuzzy one.
//
//   Keep this header free of CUDA-only types and of `__global__` so the host
//   compiler can include it. The only CUDA sprinkle is the BF_HD macro below,
//   which expands to nothing under the host compiler.
//
// WHAT DAS BEAMFORMING IS  (the science, in one breath)
//   An ultrasound probe is a row of tiny piezo ELEMENTS. To make a B-mode image
//   we fire a pulse into the body; echoes from scatterers (tissue boundaries)
//   travel back and each element records a 1-D time signal -- the "RF data".
//   A scatterer at image point P is at a slightly different distance from each
//   element, so its echo arrives at each element at a slightly different TIME.
//   Delay-and-sum says: to find the brightness at P, look up each element's RF
//   signal at exactly the round-trip travel time to P, and SUM those samples.
//   When P really holds a scatterer all the looked-up samples are the same echo
//   in phase -> they reinforce (bright). When P is empty they are random phase
//   -> they cancel (dark). That coherent sum IS the focused image.
//
//   This is mathematically the same "gather + interpolate + accumulate" shape
//   as CT backprojection (project 4.01): one independent thread per output
//   pixel, each looping over a set of inputs (there: angles; here: elements).
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cu, kernels.cuh.
// ===========================================================================
#pragma once

// ---------------------------------------------------------------------------
// BF_HD: decorate a function so it compiles for BOTH host and device under
// nvcc, and as a plain inline function under the host compiler (which has never
// heard of __host__/__device__). This is the PATTERNS.md §2 HD-macro idiom.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define BF_HD __host__ __device__
#else
#define BF_HD
#endif

#include <cstddef>   // std::size_t

// ---------------------------------------------------------------------------
// BeamformGeom: everything needed to turn raw RF samples into a focused image.
//   This struct is filled once (host) and its scalar fields are passed to the
//   kernel by value, so the device sees the identical geometry the CPU used.
//
//   Coordinate system (SI units, metres & seconds):
//     * The probe is a LINEAR array lying on the line y = 0, elements spaced
//       `pitch` apart and centred on x = 0. Element e sits at
//           (x_e, 0),  x_e = (e - (n_elements-1)/2) * pitch.
//     * The image is a rectangular grid in the (x, z) plane: x is lateral
//       (across the probe), z is depth (into the body, z >= 0). Pixel (ix, iz)
//       is at world point
//           x = x_min + ix * dx,   z = z_min + iz * dz.
//     * RF data is `rf[e * n_samples + t]` = element e's signal at sample t,
//       i.e. at time t / fs seconds after the transmit pulse left the probe.
// ---------------------------------------------------------------------------
struct BeamformGeom {
    int   n_elements = 0;   // number of transducer elements (e.g. 64)
    int   n_samples  = 0;   // RF samples recorded per element (fast-time length)
    int   nx         = 0;   // image width  (pixels, lateral)
    int   nz         = 0;   // image height (pixels, depth)

    float fs         = 0.0f;  // RF sampling frequency  [Hz]   (samples per second)
    float c          = 0.0f;  // speed of sound         [m/s]  (~1540 in soft tissue)
    float pitch      = 0.0f;  // element spacing        [m]
    float x_min      = 0.0f;  // image grid origin x    [m]
    float z_min      = 0.0f;  // image grid origin z    [m]  (>= 0, just below probe)
    float dx         = 0.0f;  // pixel pitch, lateral   [m]
    float dz         = 0.0f;  // pixel pitch, depth     [m]
    float t0         = 0.0f;  // time of the FIRST RF sample [s] (transmit offset)
};

// ---------------------------------------------------------------------------
// element_x: world x-coordinate of transducer element `e`.
//   The array is centred on x = 0, so element (n-1)/2 sits at x = 0. Pure
//   geometry; identical on host and device.
// ---------------------------------------------------------------------------
BF_HD inline float element_x(const BeamformGeom& g, int e) {
    // Centre the array: subtract the midpoint index (n-1)/2 so the array
    // straddles x = 0 symmetrically.
    return (e - 0.5f * (g.n_elements - 1)) * g.pitch;
}

// ---------------------------------------------------------------------------
// pixel_xz: world (x, z) of image pixel (ix, iz). Returned through out-params
//   because BF_HD inline must stay CUDA-only-type-free (no float2 in host code).
// ---------------------------------------------------------------------------
BF_HD inline void pixel_xz(const BeamformGeom& g, int ix, int iz,
                           float& x, float& z) {
    x = g.x_min + ix * g.dx;     // lateral position of this column
    z = g.z_min + iz * g.dz;     // depth     position of this row
}

// ---------------------------------------------------------------------------
// das_contribution: the per-(pixel,element) heart of DAS. THIS is the formula
//   the CPU loop and the GPU thread both call, guaranteeing identical results.
//
//   Physics: we assume the transmit pulse illuminates the whole field from a
//   virtual source at the array centre (a common "synthetic transmit" model
//   used for teaching). The echo from pixel P=(px,pz) recorded by element e
//   travels:
//       transmit leg : array-centre -> P,  distance = sqrt(px^2 + pz^2)
//       receive  leg : P -> element e,      distance = sqrt((px-xe)^2 + pz^2)
//   Round-trip time  tau = (d_tx + d_rx) / c. We then convert tau to a
//   fractional RF SAMPLE index and linearly interpolate element e's signal
//   there. That interpolated value is this element's contribution to P.
//
//   Returns the interpolated RF sample (0 if the delay falls outside the
//   recorded window -- the same guard on both sides keeps CPU==GPU exact).
//
//   Parameters:
//     g     : geometry (units above)
//     rf    : [n_elements * n_samples] RF data, row-major (element-major)
//     e     : element index            [0, n_elements)
//     px,pz : world coords of the pixel [m]  (z = depth > 0)
//   Complexity: O(1) -- a handful of FLOPs + one interpolated global load.
// ---------------------------------------------------------------------------
BF_HD inline float das_contribution(const BeamformGeom& g, const float* rf,
                                    int e, float px, float pz) {
    const float xe = element_x(g, e);            // element's lateral position

    // Transmit leg: distance from the (x=0) array centre down to the pixel.
    // We use a single virtual transmit from the array centre so every pixel has
    // a well-defined transmit path (full derivation in THEORY.md §"The math").
    const float d_tx = sqrtf(px * px + pz * pz);

    // Receive leg: pixel back up to this particular element.
    const float dxr = px - xe;                   // lateral gap pixel<->element
    const float d_rx = sqrtf(dxr * dxr + pz * pz);

    // Round-trip time of flight, then express it relative to the first stored
    // sample (t0) and scale by the sampling rate to get a fractional sample idx.
    const float tau = (d_tx + d_rx) / g.c;       // seconds
    const float fidx = (tau - g.t0) * g.fs;      // fractional RF sample index

    // Linear interpolation between the two bracketing integer samples. floorf
    // matches std::floor on the host, so the chosen samples are identical.
    const int   i0 = (int)floorf(fidx);
    if (i0 < 0 || i0 + 1 >= g.n_samples) {
        return 0.0f;                             // delay outside recorded window
    }
    const float frac = fidx - (float)i0;         // interpolation weight in [0,1)
    const float* sig = rf + (std::size_t)e * g.n_samples;  // this element's row
    // s = (1-frac)*sig[i0] + frac*sig[i0+1]. Written with the same operation
    // order on both sides so float rounding agrees exactly.
    return sig[i0] * (1.0f - frac) + sig[i0 + 1] * frac;
}

// ---------------------------------------------------------------------------
// das_pixel: full delay-and-sum for ONE image pixel: sum every element's
//   contribution. This is what one CPU iteration and one GPU thread each do.
//   The result is the (signed) coherent sum; main.cu takes |.| for the B-mode
//   envelope when reporting. Keeping the *signed* sum here makes CPU==GPU a
//   clean equality check before any nonlinear envelope step.
// ---------------------------------------------------------------------------
BF_HD inline float das_pixel(const BeamformGeom& g, const float* rf,
                             int ix, int iz) {
    float px, pz;
    pixel_xz(g, ix, iz, px, pz);                 // this pixel's world position
    float acc = 0.0f;                            // running coherent sum
    for (int e = 0; e < g.n_elements; ++e) {
        acc += das_contribution(g, rf, e, px, pz);
    }
    return acc;
}
