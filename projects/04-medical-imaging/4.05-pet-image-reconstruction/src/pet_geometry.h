// ===========================================================================
// src/pet_geometry.h  --  The ONE TRUE projection geometry (CPU/GPU shared)
// ---------------------------------------------------------------------------
// Project 4.5 : PET Image Reconstruction (MLEM / OS-EM)
//
// WHY THIS HEADER EXISTS  (the "HD-core" idiom -- docs/PATTERNS.md §2)
//   MLEM verification is only meaningful if the CPU reference and the GPU kernel
//   compute the SAME arithmetic. The riskiest place for CPU and GPU to silently
//   diverge is the projection geometry -- the map from image pixels to lines of
//   response (LORs) and back. So we put that geometry HERE, once, as inline
//   `__host__ __device__` functions, and BOTH sides call it:
//       * reference_cpu.cpp  (compiled by cl.exe/g++) includes this header.
//       * kernels.cu         (compiled by nvcc)       includes this header.
//   Because the source text is identical, the forward/back projection sums are
//   built from the same operations in the same order -> CPU and GPU agree to
//   within floating-point FMA rounding (see THEORY.md "How we verify").
//
// WHAT A PET SCAN MEASURES (the physics, in one paragraph)
//   A positron-emitting tracer (e.g. 18F-FDG) is injected. Each positron
//   annihilates with a nearby electron, emitting TWO 511 keV gamma photons in
//   (almost) opposite directions. A ring of detectors registers the two hits as
//   a COINCIDENCE: the annihilation happened somewhere on the straight line
//   joining the two detectors -- a "Line Of Response" (LOR). Over a scan we
//   collect a COUNT per LOR: how many annihilations fell on that line. Stacked by
//   angle, these counts form a SINOGRAM y. Reconstruction inverts y to recover
//   the 3-D (here 2-D) tracer concentration image x. See THEORY.md.
//
// THE FORWARD MODEL (what the geometry below encodes)
//   We use the same clean 2-D PARALLEL-BEAM geometry as the CT flagship (4.01):
//     * The image is an N x N grid of pixels covering the square world
//       [-W, +W] x [-W, +W].  Pixel (px,py) center is at
//           wx = -W + px*pix,  wy = -W + py*pix,   pix = 2W/(N-1).
//     * An LOR is indexed by (angle k, detector bin j). Angle theta_k = k*pi/K
//       spans 180 degrees (a parallel-beam PET rebinning). Detector bin j sits at
//       signed offset  s_j = (j - (D-1)/2) * ds  along the detector axis.
//     * The system-matrix element A[(k,j), (px,py)] is the length/weight with
//       which pixel (px,py) contributes to LOR (k,j). We use the standard
//       "pixel-driven" ray model: a pixel contributes to the LOR whose detector
//       bin its projected coordinate s = wx*cos + wy*sin falls in, split by
//       LINEAR INTERPOLATION between the two nearest bins. This is the transpose-
//       consistent pair used by forward_project (image -> sinogram) and
//       backproject (sinogram -> image): both walk pixels and split the same way,
//       so backproject is the exact transpose of forward_project. That matched
//       pair is what MLEM needs to converge (THEORY.md "The algorithm").
//
// READ THIS BEFORE: reference_cpu.h/.cpp and kernels.cuh/.cu (both include it).
// ===========================================================================
#pragma once

// -------------------------------------------------------------------------
// __host__ __device__ portability shim.
//   Under nvcc (__CUDACC__ defined) we decorate the inline helpers so they can
//   be called from BOTH host code and device kernels. Under the plain host
//   compiler those decorators don't exist, so we make them vanish. This is the
//   canonical HD-macro from docs/PATTERNS.md §2.
// -------------------------------------------------------------------------
#ifdef __CUDACC__
#define PET_HD __host__ __device__
#else
#define PET_HD
#endif

#include <cmath>   // std::floor (host); nvcc maps floor() for device too

// ---------------------------------------------------------------------------
// PetGeom: a Plain-Old-Data bundle of everything the projection math needs.
//   It is trivially copyable, so we can pass it BY VALUE into a kernel (it lands
//   in constant/param memory) and also hold it on the host. No pointers, no STL
//   -> safe to use inside a __device__ function.
//
//   Fields (units in comments):
//     N       image side length in pixels (image is N x N)
//     K       number of projection angles (LOR angle count)
//     D       number of detector bins per angle
//     ds      detector bin spacing, in world units
//     W       image half-width: world spans [-W, +W] in x and y (world units)
//   Derived (filled by make_geom so both sides compute them identically):
//     pix     world units per pixel = 2W/(N-1)
//     center  detector index of the s=0 ray = (D-1)/2
// ---------------------------------------------------------------------------
struct PetGeom {
    int    N      = 0;
    int    K      = 0;
    int    D      = 0;
    float  ds     = 0.0f;
    float  W      = 0.0f;
    float  pix    = 0.0f;   // derived
    float  center = 0.0f;   // derived
};

// make_geom: fill the derived fields once, on the host, so pix/center are bit-
// identical everywhere they are used. Guards N==1 (avoid divide-by-zero).
inline PetGeom make_geom(int N, int K, int D, float ds, float W) {
    PetGeom g;
    g.N = N; g.K = K; g.D = D; g.ds = ds; g.W = W;
    g.pix    = (N > 1) ? (2.0f * W / static_cast<float>(N - 1)) : 0.0f;
    g.center = 0.5f * static_cast<float>(D - 1);
    return g;
}

// ---------------------------------------------------------------------------
// pixel_world_x / pixel_world_y: world coordinate of a pixel center.
//   Pulled out as tiny inline helpers so the forward and back projectors -- and
//   the CPU and GPU -- all agree on exactly where pixel (px,py) sits.
// ---------------------------------------------------------------------------
PET_HD inline float pixel_world_x(const PetGeom& g, int px) {
    return -g.W + static_cast<float>(px) * g.pix;
}
PET_HD inline float pixel_world_y(const PetGeom& g, int py) {
    return -g.W + static_cast<float>(py) * g.pix;
}

// ---------------------------------------------------------------------------
// detector_fidx: the FRACTIONAL detector-bin index a pixel projects to at angle
//   k. This is the heart of the system matrix.
//     s     = wx*cos(theta_k) + wy*sin(theta_k)   (signed distance along det.)
//     fidx  = s/ds + center                        (bin coordinate, may be frac.)
//   The caller (forward or back projector) then linearly interpolates between
//   bins floor(fidx) and floor(fidx)+1 with weight (fidx - floor). Passing
//   cos/sin IN (precomputed once on the host in double->float) guarantees the
//   CPU and GPU use bit-identical trig -- cosf() on device and std::cos() on host
//   can differ in the last bit, which over K angles would break exact agreement.
// ---------------------------------------------------------------------------
PET_HD inline float detector_fidx(const PetGeom& g, float wx, float wy,
                                  float cos_k, float sin_k) {
    const float s = wx * cos_k + wy * sin_k;   // projected coordinate (world units)
    return s / g.ds + g.center;                // -> fractional bin index
}

// ---------------------------------------------------------------------------
// split_bin: given a fractional bin index, return the lower integer bin j0 and
//   the linear-interpolation weight w in [0,1) for the UPPER bin (so the lower
//   bin gets 1-w). Returns whether (j0, j0+1) are both inside [0, D). Callers
//   skip out-of-field contributions when in_range is false.
//
//   We factor this out because forward_project and backproject MUST split a
//   pixel's contribution the same way for backproject to be the transpose of
//   forward_project (required by MLEM). One function, one truth.
// ---------------------------------------------------------------------------
PET_HD inline bool split_bin(const PetGeom& g, float fidx, int& j0, float& w) {
    // floorf on device / std::floor on host: same value for finite inputs.
    const float ff = floorf(fidx);
    j0 = static_cast<int>(ff);
    w  = fidx - ff;                     // fractional part -> weight of bin j0+1
    return (j0 >= 0) && (j0 + 1 < g.D); // both neighbors must exist
}
