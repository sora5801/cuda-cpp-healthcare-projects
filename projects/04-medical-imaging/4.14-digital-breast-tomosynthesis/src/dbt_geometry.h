// ===========================================================================
// src/dbt_geometry.h  --  Shared DBT geometry + the ONE ray-sampling formula
// ---------------------------------------------------------------------------
// Project 4.14 : Digital Breast Tomosynthesis (see ../THEORY.md, catalog 4.14)
//
// WHAT THIS HEADER IS
//   The single source of truth for the *per-ray physics* of our simplified
//   limited-angle tomosynthesis. It is included by BOTH:
//     * reference_cpu.cpp  (compiled by the host C++ compiler), and
//     * kernels.cu         (compiled by nvcc),
//   so the CPU reference and the GPU kernels run *byte-for-byte identical math*.
//   This is the "shared __host__ __device__ core" idiom (docs/PATTERNS.md §2):
//   put the per-element formula in one place, loop it on the host, call it from
//   one thread on the device -> exact CPU/GPU parity, so verification is honest.
//
//   Keep this header free of CUDA-only constructs (no __global__, no <<< >>>):
//   the host compiler must be able to include it. The only CUDA sprinkle allowed
//   is the __host__ __device__ decorator, hidden behind the DBT_HD macro below.
//
// THE PHYSICS IN ONE SENTENCE
//   An X-ray projection is a set of *line integrals* of the object's linear
//   attenuation. Reconstruction inverts that: given the line integrals from a
//   few angles, recover the 2-D attenuation image. DBT does this from only a
//   NARROW angular range (~15-50 deg), which is what makes it "limited-angle".
//
// WHY LIMITED ANGLE IS HARD (the whole point of DBT)
//   With a full 180 deg scan, Filtered BackProjection (project 4.01) inverts the
//   Radon transform cleanly. With only a +/-25 deg wedge, the data are missing a
//   huge cone in Fourier space (the "missing wedge"), so FBP is unstable and
//   smears structures along depth. Iterative algebraic methods (SART/OS-EM) that
//   repeatedly *forward-project* a running estimate and correct it from the
//   residual behave far better under this ill-posed geometry. This project
//   implements SART; THEORY.md derives why.
//
// COORDINATE CONVENTIONS (used identically by CPU and GPU)
//   * The image is N x N pixels covering world square [-W, W]^2 (W = world_half).
//     Pixel (px,py) sits at world (x,y) = (-W + px*pix, -W + py*pix),
//     with pix = 2W/(N-1). image[py*N + px] is its attenuation value.
//   * A projection angle theta tilts the parallel X-ray beam. For a ray we use
//     the standard Radon parameterisation: a ray at angle theta and signed
//     detector offset s is the line { (x,y) : x*cos(theta) + y*sin(theta) = s }.
//     Detector bin j maps to s_j = (j - (n_det-1)/2) * ds.
//   * DBT angles span a NARROW symmetric wedge: theta_k in [-half_span, +half_span]
//     (radians), n_angles of them. reference_cpu.cpp::compute_angles() builds the
//     cos/sin tables once so CPU and GPU share bit-identical trig.
//
// READ THIS BEFORE: reference_cpu.h, kernels.cuh (both build on these symbols).
// ===========================================================================
#pragma once

#include <cstddef>   // std::size_t

// ---------------------------------------------------------------------------
// DBT_HD: expands to "__host__ __device__" under nvcc (so the inline helpers
// below can run on BOTH the CPU and inside a kernel), and to nothing under a
// plain host compiler (which has never heard of those decorators). This is the
// HD-macro idiom from docs/PATTERNS.md §2 -- the key to CPU/GPU numeric parity.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define DBT_HD __host__ __device__
#else
#define DBT_HD
#endif

// ---------------------------------------------------------------------------
// clampi: clamp an integer index into [lo, hi]. Used when a ray steps just past
// the image border so we sample the edge pixel instead of reading out of bounds.
// (Constexpr-friendly, branch-cheap; identical on host and device.)
// ---------------------------------------------------------------------------
DBT_HD inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// ---------------------------------------------------------------------------
// bilinear_sample: read image[] at a *fractional* pixel coordinate (fx, fy)
// using 2x2 bilinear interpolation. This is THE interpolation both the forward
// projector (sampling the image along a ray) and any resampling step share, so
// it lives here and is called identically from host and device.
//
//   image : N x N attenuation image, row-major, image[iy*N + ix].
//   fx,fy : fractional pixel coordinates (0 = first pixel centre, N-1 = last).
//   returns the interpolated attenuation; out-of-range coords clamp to the edge
//           (equivalent to assuming zero-gradient air outside the breast).
//
// Bilinear (not nearest) matters: SART's forward/back projector must be each
// other's transpose-ish adjoint for the iteration to converge smoothly; smooth
// interpolation avoids the aliasing streaks nearest-neighbour would inject.
// ---------------------------------------------------------------------------
DBT_HD inline float bilinear_sample(const float* image, int N, float fx, float fy) {
    // Integer lower-left corner of the 2x2 neighbourhood.
    const int ix = (int)floorf(fx);
    const int iy = (int)floorf(fy);
    // Fractional weights within the cell (0..1).
    const float tx = fx - (float)ix;
    const float ty = fy - (float)iy;
    // Clamp the four sample indices so border rays never read out of bounds.
    const int ix0 = clampi(ix,     0, N - 1);
    const int ix1 = clampi(ix + 1, 0, N - 1);
    const int iy0 = clampi(iy,     0, N - 1);
    const int iy1 = clampi(iy + 1, 0, N - 1);
    // Fetch the four corners.
    const float v00 = image[(std::size_t)iy0 * N + ix0];
    const float v01 = image[(std::size_t)iy0 * N + ix1];
    const float v10 = image[(std::size_t)iy1 * N + ix0];
    const float v11 = image[(std::size_t)iy1 * N + ix1];
    // Interpolate along x on both rows, then along y between them.
    const float a = v00 * (1.0f - tx) + v01 * tx;   // bottom row
    const float b = v10 * (1.0f - tx) + v11 * tx;   // top row
    return a * (1.0f - ty) + b * ty;
}

// ---------------------------------------------------------------------------
// forward_ray_integral: the CORE FORWARD PROJECTION for one ray.
//
//   Compute the line integral of `image` along the ray that belongs to
//   projection angle k (cosine ck = cos(theta_k), sine sk = sin(theta_k)) and
//   detector bin j. This is a single element of the *system matrix times image*
//   product A*x that SART needs every iteration.
//
//   Geometry: the ray is the set of world points p(t) whose signed distance to
//   the beam axis equals the detector offset s_j. We march ALONG the ray in
//   n_steps equal steps across the image extent, bilinearly sampling the image
//   and accumulating (a midpoint Riemann sum of the line integral). The step
//   length in world units is baked into `ray_len / n_steps` by the caller when
//   it scales the result -- here we return the *mean* sample so the caller can
//   multiply by the physical ray length once (keeps this function branch-light).
//
//   Parameters:
//     image      : N x N attenuation image (the current estimate).
//     N          : image side length in pixels.
//     ck, sk     : cos/sin of this projection angle.
//     s          : signed detector offset of this bin (world units).
//     W          : world half-extent (image covers [-W, W]^2).
//     pix        : world units per pixel = 2W/(N-1).
//     n_steps    : number of samples taken along the ray (fixed per problem).
//   Returns: the SUM of the n_steps bilinear samples (caller multiplies by the
//            per-step world length to get the physical line integral). Returning
//            the raw sum keeps forward and back projection exact adjoints.
//
//   Direction vectors:
//     The ray at angle theta has NORMAL (cos,sin) and DIRECTION (-sin,cos).
//     A point on the ray is:  s*(cos,sin) + t*(-sin,cos), for t in [-L, L].
//     We convert world (x,y) to fractional pixel coords via
//     fx = (x + W)/pix, fy = (y + W)/pix, then bilinear_sample().
// ---------------------------------------------------------------------------
DBT_HD inline float forward_ray_integral(const float* image, int N,
                                         float ck, float sk, float s,
                                         float W, float pix, int n_steps) {
    // Half-length of the chord we march: the image diagonal half-extent, so the
    // ray is sampled across the whole image no matter its angle. sqrt(2)*W.
    const float L = 1.41421356f * W;
    // t ranges over [-L, +L] in (n_steps) samples; dt is the parametric step.
    const float dt = (n_steps > 1) ? (2.0f * L / (n_steps - 1)) : 0.0f;
    float acc = 0.0f;
    for (int m = 0; m < n_steps; ++m) {
        const float t = -L + m * dt;            // position along the ray
        // World coordinates of this sample: s*normal + t*direction.
        const float x = s * ck - t * sk;        // = s*cos + t*(-sin)
        const float y = s * sk + t * ck;        // = s*sin + t*( cos)
        // World -> fractional pixel coordinates.
        const float fx = (x + W) / pix;
        const float fy = (y + W) / pix;
        // Only accumulate samples that fall inside the image footprint; outside
        // is air (zero attenuation) and contributes nothing.
        if (fx >= -0.5f && fx <= N - 0.5f && fy >= -0.5f && fy <= N - 0.5f) {
            acc += bilinear_sample(image, N, fx, fy);
        }
    }
    return acc;   // caller scales by the per-step world length (dt) once
}
