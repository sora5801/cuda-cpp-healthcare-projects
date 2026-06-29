// ===========================================================================
// src/kmeans.h  --  Shared (host + device) k-means primitives
// ---------------------------------------------------------------------------
// Project 11.09 : Flow Cytometry & High-Content Screening Analysis
//
// WHAT THIS PROJECT COMPUTES
//   k-means clustering of flow-cytometry EVENTS. Each event is a cell measured
//   on D markers (a D-dimensional point); we partition N events into K clusters
//   (cell populations). k-means alternates two steps until convergence:
//     ASSIGN : each point -> its nearest centroid (Euclidean).   [parallel/point]
//     UPDATE : each centroid <- mean of its assigned points.      [a REDUCTION]
//
// WHY A GPU
//   Modern sorters produce ~10^5 cells/second at 20-50 parameters; clustering
//   millions of events is the bottleneck. ASSIGN is embarrassingly parallel (one
//   thread per event). UPDATE is a scatter-reduction: many threads accumulate
//   into the same K centroids -> atomicAdd. This flagship's lesson is the
//   ATOMIC REDUCTION, made DETERMINISTIC with fixed-point integers (below).
//
// DETERMINISM TRICK (same idea as project 5.01)
//   Float atomicAdd is order-dependent (non-associative) -> non-reproducible.
//   We normalize events to [0,1] and accumulate coordinates in FIXED-POINT
//   integers (atomicAdd on unsigned long long), which commute -> the GPU result
//   is reproducible AND equals the CPU result exactly.
//
//   These helpers are __host__ __device__ (KM_HD) so the CPU reference and the
//   GPU kernels share identical math.
// ===========================================================================
#pragma once

#include <cstdint>

#ifdef __CUDACC__
#define KM_HD __host__ __device__
#else
#define KM_HD
#endif

// Fixed-point scale: coordinates in [0,1] are stored as integers in [0, KM_SCALE].
// 2^20 ~ 1e6 gives ~6 significant digits; sums of N=millions stay well within
// unsigned long long.
static const unsigned long long KM_SCALE = 1ull << 20;

// Quantize a normalized coordinate (assumed in [0,1]) to fixed-point.
KM_HD inline unsigned long long km_to_fixed(float v) {
    return static_cast<unsigned long long>(static_cast<double>(v) * KM_SCALE);
}

// Squared Euclidean distance between a D-dim point and a centroid.
KM_HD inline double km_sqdist(const float* point, const float* centroid, int D) {
    double s = 0.0;
    for (int d = 0; d < D; ++d) {
        const double diff = static_cast<double>(point[d]) - static_cast<double>(centroid[d]);
        s += diff * diff;
    }
    return s;
}

// Index of the nearest centroid to `point` (ties -> lowest index, via strict <).
KM_HD inline int km_nearest(const float* point, const float* centroids, int K, int D) {
    int best = 0;
    double best_d = km_sqdist(point, centroids, D);
    for (int k = 1; k < K; ++k) {
        const double d = km_sqdist(point, centroids + static_cast<std::size_t>(k) * D, D);
        if (d < best_d) { best_d = d; best = k; }
    }
    return best;
}
