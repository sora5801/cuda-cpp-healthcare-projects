// ===========================================================================
// src/mc4dct.h  --  Shared __host__ __device__ core for 4D-CT reconstruction
// ---------------------------------------------------------------------------
// Project 4.19 : Motion-Compensated 4D-CT Reconstruction  (reduced-scope,
//                2-D teaching version -- see THEORY.md "Where this sits in the
//                real world" for what full 4D-CBCT/MCR does beyond this).
//
// WHY THIS HEADER EXISTS (the HD-macro idiom, PATTERNS.md section 2)
//   The CPU reference (reference_cpu.cpp, compiled by cl.exe) and the GPU kernel
//   (kernels.cu, compiled by nvcc) must produce BYTE-FOR-BYTE identical images so
//   verification is exact, not fuzzy. The only way to guarantee that is to write
//   the per-pixel physics ONCE, in inline functions marked `__host__ __device__`,
//   and call the SAME functions from both sides. That is what this header is.
//
//   Keep this header free of CUDA-only constructs (no __global__, no <<<>>>), so
//   the plain host compiler can include it too. Only the HD decorator differs.
//
// THE SCIENCE IN ONE PARAGRAPH
//   In a chest CT the patient breathes, so the anatomy MOVES while the scanner
//   spins. 4D-CT copes by tagging each X-ray projection with the breathing
//   PHASE it was taken in (0 = full inhale ... P-1), then binning the ~thousands
//   of projections into P phase groups. Reconstructing each group on its own
//   gives P images -- but each group has few angles, so each image is streaky
//   and under-sampled. MOTION-COMPENSATED reconstruction fixes this: given a
//   Deformation Vector Field (DVF) that says how every pixel moved from a chosen
//   REFERENCE phase into each other phase, we can WARP every projection's
//   contribution back into the reference frame. Now ALL projections from ALL
//   phases reconstruct ONE sharp reference image -- motion turned from a curse
//   into free extra angles.
//
// WHAT WE MODEL HERE (honest, reduced scope)
//   * 2-D parallel-beam geometry (like flagship 4.01), not 3-D cone-beam.
//   * A KNOWN, analytic DVF (a smooth breathing warp) instead of estimating it
//     by deformable registration. Real MCR alternates reconstruction with DVF
//     estimation (Demons/optical-flow); here the DVF is given so the teaching
//     point -- the motion-compensated backprojection gather -- is isolated.
//   * We compare, on the SAME data, the naive "average of per-phase FBP" against
//     the motion-compensated reconstruction, so the learner SEES motion blur
//     disappear.
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cu, main.cu.
// ===========================================================================
#pragma once

// ---------------------------------------------------------------------------
// The HD ("host/device") portability macro (PATTERNS.md section 2).
//   * Under nvcc (__CUDACC__ defined) it expands to `__host__ __device__`, so
//     the function is compiled for BOTH the CPU and the GPU.
//   * Under the plain host compiler those keywords do not exist, so it expands
//     to nothing and the function is an ordinary inline host function.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define MC_HD __host__ __device__
#else
#define MC_HD
#endif

#include <cmath>   // std::sin, std::cos, std::floor, std::sqrt (host side)

// pi as a double literal; every angle formula below uses it.
#ifndef MC_PI
#define MC_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// Geometry constants shared by data generation, CPU, and GPU.
//   These are compile-time so the CPU and GPU agree on every derived quantity
//   (pixel size, detector center) without passing extra parameters around.
//   NOTE: the sample's header MUST match these (main.cu asserts it) so the
//   committed data and the code never silently drift apart.
// ---------------------------------------------------------------------------
struct Geom {
    int   img;          // image side length in pixels (square img x img)
    int   n_det;        // detector bins per projection
    int   n_phases;     // number of breathing phases P
    int   n_ang_phase;  // projection angles PER phase (few -> under-sampled)
    float ds;           // detector bin spacing (world units)
    float world_half;   // image spans [-world_half, world_half]^2 (world units)
    float amp;          // breathing motion amplitude (world units) -- DVF scale
};

// ---------------------------------------------------------------------------
// Portable trig / floor: on the device we want the SAME rounding as the host so
//   results match bit-for-bit. We therefore do the math in DOUBLE and cast once
//   -- nvcc maps std::cos/std::sin/std::floor(double) to the device double
//   routines, which agree with the host. We never use the fast-but-divergent
//   single-precision intrinsics (cosf/__cosf), which would break CPU==GPU parity.
//   Declared FIRST because the functions below call them.
// ---------------------------------------------------------------------------
MC_HD inline float cosf_portable(float x)   { return (float)cos((double)x); }
MC_HD inline float sinf_portable(float x)   { return (float)sin((double)x); }
MC_HD inline float floorf_portable(float x) { return (float)floor((double)x); }

// ---------------------------------------------------------------------------
// phase_motion(p, P): the breathing "position" of phase p, in [0, 1].
//   p = 0 is end-inhale (the REFERENCE phase, zero motion by construction);
//   the diaphragm then pushes down and returns over the cycle. We model the
//   cycle with a raised cosine so motion is smooth and periodic:
//       m(p) = 0.5 * (1 - cos(2*pi*p/P))   in [0,1], m(0)=0, m(P/2)=1.
//   This single scalar drives the whole DVF, so CPU and GPU share it exactly.
// ---------------------------------------------------------------------------
MC_HD inline float phase_motion(int p, int n_phases) {
    // 2*pi*p/P sweeps a full breathing cycle across the P phases.
    const float ang = (float)(2.0 * MC_PI) * (float)p / (float)n_phases;
    // Raised cosine: 0 at p=0 (reference), 1 at the opposite phase.
    return 0.5f * (1.0f - cosf_portable(ang));
}

// ---------------------------------------------------------------------------
// dvf_at(gx, geom, p): the Deformation Vector Field.
//   Given a REFERENCE-frame world point (gx.x, gx.y) and a phase p, return the
//   displacement (dx, dy) that maps the reference point to WHERE THAT TISSUE IS
//   during phase p. In real 4D-CT this field is estimated by deformable image
//   registration; here we prescribe a smooth, physically-plausible analytic
//   breathing warp so the teaching stays about the GPU gather, not registration.
//
//   Our warp: mostly a vertical push (the diaphragm moves head-foot), scaled by
//   the phase motion m(p) and by depth (more motion lower in the chest, less at
//   the top), plus a gentle expansion so it is a genuine NON-RIGID field:
//       dy =  amp * m(p) * (0.5 - 0.5*ny)      (ny in [-1,1]: bottom moves most)
//       dx =  0.25 * amp * m(p) * nx           (slight left-right expansion)
//   where (nx, ny) are normalized coordinates in [-1,1]. m(0)=0 => phase 0 is the
//   undeformed reference, exactly as the reconstruction assumes.
// ---------------------------------------------------------------------------
struct Vec2 { float x, y; };

MC_HD inline Vec2 dvf_at(float wx, float wy, const Geom& g, int p) {
    const float m  = phase_motion(p, g.n_phases);          // breathing amount [0,1]
    // Normalize world coords to [-1,1] so the field shape is resolution-free.
    const float nx = (g.world_half > 0.f) ? wx / g.world_half : 0.f;
    const float ny = (g.world_half > 0.f) ? wy / g.world_half : 0.f;
    Vec2 d;
    // Vertical diaphragm-like push: larger toward the bottom (ny -> -1).
    d.y = g.amp * m * (0.5f - 0.5f * ny);
    // Small lateral expansion so the field is non-rigid (not a pure translation).
    d.x = 0.25f * g.amp * m * nx;
    return d;
}

// ---------------------------------------------------------------------------
// sample_projection(row, n_det, s, ds, center): linear interpolation of a 1-D
//   projection at continuous detector offset s.
//   Backprojection needs the projection value where a ray crosses the detector;
//   that crossing rarely lands on an integer bin, so we linearly interpolate
//   between the two nearest bins. Out-of-range rays contribute 0. This exact
//   routine runs on both CPU and GPU -> identical arithmetic.
// ---------------------------------------------------------------------------
MC_HD inline float sample_projection(const float* row, int n_det,
                                     float s, float ds, float center) {
    const float fidx = s / ds + center;        // fractional detector index
    const float ff   = floorf_portable(fidx);
    const int   j0   = (int)ff;                 // lower bin
    if (j0 < 0 || j0 + 1 >= n_det) return 0.0f; // ray misses the detector
    const float w = fidx - ff;                  // interpolation weight in [0,1)
    return row[j0] * (1.0f - w) + row[j0 + 1] * w;
}

// ---------------------------------------------------------------------------
// pixel_world(px, py, geom, &wx, &wy): map integer pixel (px,py) to world (x,y).
//   Pixels are centered on a grid spanning [-W, W]^2. Shared so CPU and GPU
//   place pixels identically.
// ---------------------------------------------------------------------------
MC_HD inline void pixel_world(int px, int py, const Geom& g, float* wx, float* wy) {
    const float pix = (g.img > 1) ? (2.0f * g.world_half / (g.img - 1)) : 0.0f;
    *wx = -g.world_half + px * pix;
    *wy = -g.world_half + py * pix;
}

// ---------------------------------------------------------------------------
// mc_pixel(px, py, geom, angles, cosv, sinv, filtered, motion_comp): the ONE
//   true per-pixel reconstruction formula. This is the heart of the project and
//   is called identically from the CPU loop and the GPU kernel.
//
//   For output pixel (px,py):
//     1. Find its reference-frame world position (wx, wy).
//     2. For every phase p and every angle within that phase:
//          - if motion_comp: displace the pixel by the DVF for phase p, so we
//            sample the projection WHERE THIS TISSUE ACTUALLY WAS during phase p.
//            (wx', wy') = (wx,wy) + dvf_at(wx,wy,p).
//          - else (naive): use (wx,wy) unchanged -> motion smears the image.
//          - project onto the detector: s = wx'*cos(theta) + wy'*sin(theta),
//            interpolate the ramp-filtered projection there, accumulate.
//     3. Scale by d(theta) = pi / (total number of angles) so the value is a
//        proper backprojection integral.
//
//   The ONLY difference between naive and motion-compensated is the DVF shift --
//   which is exactly the didactic point.
// ---------------------------------------------------------------------------
MC_HD inline float mc_pixel(int px, int py, const Geom& g,
                            const float* cosv, const float* sinv,
                            const float* filtered,   // [P*n_ang_phase * n_det]
                            int   motion_comp) {
    float wx, wy;
    pixel_world(px, py, g, &wx, &wy);
    const float center = 0.5f * (g.n_det - 1);          // detector index of s=0

    float acc = 0.0f;
    // Loop phases (outer) then the few angles within each phase (inner).
    for (int p = 0; p < g.n_phases; ++p) {
        // Reference->phase displacement of THIS pixel (0 if motion_comp off).
        float sx = wx, sy = wy;
        if (motion_comp) {
            const Vec2 d = dvf_at(wx, wy, g, p);
            sx = wx + d.x;
            sy = wy + d.y;
        }
        for (int a = 0; a < g.n_ang_phase; ++a) {
            const int k = p * g.n_ang_phase + a;        // global angle index
            const float s = sx * cosv[k] + sy * sinv[k];
            const float* row = filtered + (long long)k * g.n_det;
            acc += sample_projection(row, g.n_det, s, g.ds, center);
        }
    }
    // d(theta): angles are spread over [0, pi) across ALL phases combined.
    const int total_ang = g.n_phases * g.n_ang_phase;
    const float scale = (float)MC_PI / (total_ang > 0 ? total_ang : 1);
    return acc * scale;
}
