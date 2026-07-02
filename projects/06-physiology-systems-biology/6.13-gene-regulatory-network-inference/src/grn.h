// ===========================================================================
// src/grn.h  --  The shared __host__ __device__ mutual-information CORE
// ---------------------------------------------------------------------------
// Project 6.13 : Gene Regulatory Network Inference (ARACNE, MI + DPI)
//
// WHY THIS HEADER EXISTS  (PATTERNS.md sec 2, the "HD-macro idiom")
//   The per-pair PHYSICS -- how a joint histogram becomes a mutual-information
//   score in nats -- must be *identical* on the CPU reference and the GPU
//   kernel, or verification is meaningless. So we put that one true formula in
//   ONE header, decorated `__host__ __device__` when compiled by nvcc and left
//   undecorated when compiled by the plain host compiler. Then:
//       * reference_cpu.cpp  (host compiler)  loops mi_from_joint() over pairs;
//       * kernels.cu         (nvcc)           calls mi_from_joint() from a thread.
//   Same integer counts in, same double-precision log arithmetic out -> the two
//   results agree to ~1e-12 (THEORY.md sec "How we verify correctness").
//
// WHAT ARACNE DOES  (the science; full derivation in ../THEORY.md)
//   A gene regulatory network (GRN) is a graph: an edge A--B means the
//   expression of gene A statistically depends on gene B (a transcription
//   factor regulating a target, say). ARACNE scores every unordered gene PAIR
//   by their MUTUAL INFORMATION -- a nonlinear, distribution-free measure of
//   statistical dependence:
//       I(A;B) = sum_{a,b} p(a,b) * ln( p(a,b) / (p(a) p(b)) )   [nats]
//   I = 0 exactly iff A and B are independent; larger I = stronger coupling.
//   We estimate p(.) by DISCRETIZING each gene's expression into B equal-width
//   bins and counting a joint B x B histogram over the S samples (cells).
//
//   ARACNE then applies the DATA PROCESSING INEQUALITY (DPI) to remove
//   INDIRECT edges: in a chain A -> C -> B, the DPI guarantees
//   I(A;B) <= min(I(A;C), I(C;B)), so the weakest edge of every triangle is the
//   likely indirect one and is pruned. That pruning lives in kernels/reference;
//   this header supplies the MI those steps consume.
//
// LAYOUT CONVENTION (used everywhere)
//   Expression matrix is GENES x SAMPLES, row-major: gene g, sample s lives at
//   expr[g * n_samples + s]. Discretized copy `disc` has the same layout but
//   holds a bin index in [0, n_bins). The MI matrix is a dense G x G array,
//   symmetric, with a zero diagonal.
//
// READ THIS BEFORE: reference_cpu.h, kernels.cuh.  (This header is pure -- no
// CUDA types, no __global__ -- so the host compiler can include it too.)
// ===========================================================================
#pragma once

#include <cstdint>

// ---------------------------------------------------------------------------
// GRN_HD : the portable "runs on host AND device" decorator.
//   Under nvcc (__CUDACC__ defined) it expands to `__host__ __device__`, so the
//   SAME function is compiled into both a CPU version and a device version.
//   Under the plain host compiler those keywords do not exist, so it expands to
//   nothing. This is the single idiom that guarantees CPU/GPU numeric parity.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define GRN_HD __host__ __device__
#else
#define GRN_HD
#endif

// ---------------------------------------------------------------------------
// logval : the natural logarithm, resolved to the SAME implementation on both
// sides. Under nvcc a call to ::log(double) inside a __host__ __device__
// function compiles to CUDA's device log() when generating device code and to
// the C runtime log() when generating host code -- both are correctly-rounded
// IEEE-754 double log, so they agree to the last ~1 ULP. We include <cmath>
// (host) / <math.h> semantics via the global ::log. Wrapping it in one place
// documents the single transcendental that our MI depends on.
// ---------------------------------------------------------------------------
#include <cmath>
GRN_HD inline double logval(double x) { return log(x); }

// Discretization width. B = 8 bins is a common ARACNE default for a few hundred
// samples: enough resolution to see structure, coarse enough that each of the
// B*B joint cells still collects several counts (avoiding an all-noise
// histogram). It is a COMPILE-TIME constant so the B*B joint histogram fits in
// a fixed-size per-thread stack array (see kernels.cu) and both sides use the
// exact same binning. THEORY.md sec "Numerical considerations" discusses the
// bias/variance trade-off and adaptive (equi-quantile) binning as an exercise.
#ifndef GRN_N_BINS
#define GRN_N_BINS 8
#endif
constexpr int N_BINS = GRN_N_BINS;              // bins per gene
constexpr int JOINT_CELLS = N_BINS * N_BINS;    // cells in the B x B joint table

// ---------------------------------------------------------------------------
// discretize_value : map one real expression value to a bin index in [0, B).
//   Equal-width binning between the gene's own [lo, hi] range. We clamp so the
//   maximum value (which would otherwise map to B) lands in the last valid bin,
//   and guard the degenerate lo==hi (constant gene) case by returning bin 0.
//   Because BOTH sides call this identical routine on identical inputs, the two
//   discretized matrices are bit-identical -> the joint counts are identical ->
//   the only floating-point work left is the log sum in mi_from_joint().
//
//   value : the raw expression level (arbitrary units; synthetic here)
//   lo,hi : this gene's min and max over all samples (its dynamic range)
//   returns: an integer bin in [0, N_BINS)
// ---------------------------------------------------------------------------
GRN_HD inline int discretize_value(double value, double lo, double hi) {
    if (hi <= lo) return 0;                       // constant gene -> single bin
    double t = (value - lo) / (hi - lo);          // normalize to [0, 1]
    int bin = static_cast<int>(t * N_BINS);       // scale to [0, B]
    if (bin < 0) bin = 0;                         // guard round-down below lo
    if (bin >= N_BINS) bin = N_BINS - 1;          // fold the top edge into bin B-1
    return bin;
}

// ---------------------------------------------------------------------------
// mi_from_joint : the ONE TRUE FORMULA. Given the integer joint histogram of a
// gene pair (row-major, joint[a*B + b] = # samples where gene X is in bin a and
// gene Y is in bin b) and the sample count S, return the plug-in (maximum-
// likelihood) mutual information estimate in NATS:
//
//     Ihat = (1/S) * sum_{a,b} n_ab * ln( (n_ab * S) / (r_a * c_b) )
//
//   which is the algebraic rearrangement of sum p_ab ln(p_ab/(p_a p_b)) with
//   p_ab = n_ab/S, p_a = r_a/S, p_b = c_b/S. Working from integer counts (row
//   sums r_a, column sums c_b) keeps every quantity exact until the single ln,
//   so CPU and GPU differ only in the last bits of that transcendental.
//
//   Empty cells (n_ab == 0) contribute 0 (the limit x ln x -> 0), so we skip
//   them -- which also avoids ln(0). MI is >= 0 by construction; tiny negative
//   round-off is clamped to 0.
//
//   joint : [JOINT_CELLS] integer counts, row-major (X bin outer, Y bin inner)
//   n_samples : S, the number of samples (cells) that were binned
//   returns: mutual information in nats (double), >= 0
// ---------------------------------------------------------------------------
GRN_HD inline double mi_from_joint(const int* joint, int n_samples) {
    if (n_samples <= 0) return 0.0;

    // Marginals: row sums r_a = sum_b n_ab, column sums c_b = sum_a n_ab.
    // We recompute them from the joint so the caller only has to fill `joint`.
    int row[N_BINS];    // r_a : how many samples put gene X in bin a
    int col[N_BINS];    // c_b : how many samples put gene Y in bin b
    for (int k = 0; k < N_BINS; ++k) { row[k] = 0; col[k] = 0; }
    for (int a = 0; a < N_BINS; ++a) {
        for (int b = 0; b < N_BINS; ++b) {
            int n_ab = joint[a * N_BINS + b];
            row[a] += n_ab;
            col[b] += n_ab;
        }
    }

    const double S = static_cast<double>(n_samples);
    double mi = 0.0;
    for (int a = 0; a < N_BINS; ++a) {
        if (row[a] == 0) continue;                // whole row empty -> no terms
        for (int b = 0; b < N_BINS; ++b) {
            int n_ab = joint[a * N_BINS + b];
            if (n_ab == 0) continue;              // empty cell contributes 0
            // term = p_ab * ln( p_ab / (p_a p_b) )
            //      = (n_ab/S) * ln( (n_ab * S) / (r_a * c_b) )
            double numerator   = static_cast<double>(n_ab) * S;      // n_ab * S
            double denominator = static_cast<double>(row[a]) *       // r_a * c_b
                                  static_cast<double>(col[b]);
            mi += (static_cast<double>(n_ab) / S) *
                  logval(numerator / denominator);
        }
    }
    if (mi < 0.0) mi = 0.0;                        // clamp round-off noise
    return mi;
}
