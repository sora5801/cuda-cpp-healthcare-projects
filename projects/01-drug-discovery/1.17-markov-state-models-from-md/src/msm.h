// ===========================================================================
// src/msm.h  --  Shared (host + device) Markov-State-Model primitives
// ---------------------------------------------------------------------------
// Project 1.17 : Markov State Models from MD
//
// WHAT THIS PROJECT COMPUTES
//   A Markov State Model (MSM) turns a long molecular-dynamics (MD) trajectory
//   -- a time series of conformations -- into a SMALL stochastic model of the
//   molecule's kinetics. The pipeline (reduced-scope teaching version):
//
//     (1) FEATURIZE : each MD frame is already a low-dimensional feature point
//                     (e.g. a few collective coordinates / tICA components).
//                     We treat the committed sample as that feature matrix:
//                     N frames x D features, normalized to [0,1].
//     (2) CLUSTER   : k-means partitions the N feature points into K
//                     "microstates" (a discretization of conformational space).
//                     -> ASSIGN each frame to its nearest centroid  [parallel]
//                     -> UPDATE each centroid to the mean of its frames [reduce]
//     (3) COUNT     : at a lag time tau, count transitions s(t) -> s(t+tau)
//                     between microstates, building a K x K count matrix C.
//                     -> a SCATTER of (from,to) pairs into C            [reduce]
//     (4) ESTIMATE  : row-normalize C into a transition probability matrix T
//                     (the maximum-likelihood MSM), T[i][j] = C[i][j]/sum_j.
//     (5) ANALYZE   : the stationary distribution pi (pi T = pi) gives each
//                     microstate's equilibrium population, and the second
//                     eigenvalue lambda_2 gives the slowest implied timescale
//                     t_2 = -tau / ln(lambda_2) -- the molecule's slowest
//                     kinetic process (e.g. folding / a conformational switch).
//
// WHY A GPU
//   Real MSMs are built from MILLIONS of frames (aggregated micro-to-milli-
//   seconds of GPU MD). Two steps dominate and are embarrassingly parallel:
//     * k-means ASSIGN  -- one thread per frame finds its nearest centroid.
//     * transition COUNT-- one thread per frame scatters one (from,to) pair
//       into the K x K count matrix via atomicAdd.
//   Both are exactly the "parallel-assign + atomic-reduce" pattern of flagship
//   11.09 (k-means) -- see docs/PATTERNS.md row "clustering / centroid accum."
//
// DETERMINISM TRICK (same idea as flagships 5.01 and 11.09)
//   Floating-point atomicAdd is order-dependent (non-associative) -> not
//   reproducible. We avoid it entirely on the hot paths:
//     * the transition COUNT accumulates into UNSIGNED INTEGERS -- integer adds
//       commute, so the GPU count matrix is reproducible AND equals the CPU's
//       exactly, frame for frame.
//     * the centroid UPDATE accumulates coordinates in FIXED-POINT integers
//       (km_to_fixed below), the same trick as 11.09, so GPU and CPU centroids
//       match bit-for-bit.
//   The only floating-point work that differs between CPU and GPU is the tiny
//   host-side eigen-analysis in step (5), which both sides run on IDENTICAL T,
//   so they agree to machine precision.
//
//   These helpers are __host__ __device__ (MSM_HD) so the CPU reference and the
//   GPU kernels share identical math. Keep CUDA-only constructs (__global__,
//   kernel launches) OUT of this header so the host compiler can include it.
//
// READ THIS AFTER: nothing -- start here, it is the core. Then reference_cpu.h
//   (the CPU pipeline) and kernels.cuh (the GPU twin).
// ===========================================================================
#pragma once

#include <cstddef>   // std::size_t
#include <cstdint>   // fixed-width integer types

// MSM_HD expands to "__host__ __device__" when this header is compiled by nvcc
// (which defines __CUDACC__), and to nothing when compiled by the plain host
// compiler (cl.exe / g++) for reference_cpu.cpp. One source of truth, two
// compilers, identical math -> exact CPU/GPU verification.
#ifdef __CUDACC__
#define MSM_HD __host__ __device__
#else
#define MSM_HD
#endif

// ---------------------------------------------------------------------------
// Fixed-point scale for the centroid UPDATE accumulation.
//   Feature coordinates live in [0,1]. We store each as an integer in
//   [0, MSM_SCALE] so that summing them is integer addition (commutative ->
//   deterministic). 2^20 ~ 1.05e6 gives ~6 significant digits; a sum over
//   N <= a few million frames stays far below the 2^64 range of unsigned long
//   long, so there is no overflow.
// ---------------------------------------------------------------------------
static const unsigned long long MSM_SCALE = 1ull << 20;

// Quantize a normalized coordinate (assumed in [0,1]) to fixed-point integer.
//   We cast through double first so the multiply is exact for the values we
//   use; the truncation is identical on host and device (no rounding-mode
//   divergence), which is what keeps CPU and GPU centroids equal.
MSM_HD inline unsigned long long km_to_fixed(float v) {
    return static_cast<unsigned long long>(static_cast<double>(v) * MSM_SCALE);
}

// ---------------------------------------------------------------------------
// km_sqdist: squared Euclidean distance between a D-dim frame and a centroid.
//   We accumulate in double (not float) so the SAME rounding happens on host
//   and device; squared distance (not the sqrt) is enough to rank nearest
//   centroids and is cheaper. `point` and `centroid` are length-D arrays.
// ---------------------------------------------------------------------------
MSM_HD inline double km_sqdist(const float* point, const float* centroid, int D) {
    double s = 0.0;
    for (int d = 0; d < D; ++d) {
        const double diff = static_cast<double>(point[d]) - static_cast<double>(centroid[d]);
        s += diff * diff;   // sum of squared per-feature differences
    }
    return s;
}

// ---------------------------------------------------------------------------
// km_nearest: index of the centroid closest to `point` (the microstate the
//   frame is assigned to). Ties resolve to the LOWEST index because we only
//   replace `best` on a STRICT improvement (d < best_d) -- this exact tie rule
//   must match on host and device or a frame could land in a different
//   microstate on each, breaking verification.
//   `centroids` is a flat [K*D] row-major array (centroid k starts at k*D).
// ---------------------------------------------------------------------------
MSM_HD inline int km_nearest(const float* point, const float* centroids, int K, int D) {
    int best = 0;
    double best_d = km_sqdist(point, centroids, D);   // distance to centroid 0
    for (int k = 1; k < K; ++k) {
        const double d = km_sqdist(point, centroids + static_cast<std::size_t>(k) * D, D);
        if (d < best_d) { best_d = d; best = k; }     // strict < -> lowest-index tie
    }
    return best;
}
