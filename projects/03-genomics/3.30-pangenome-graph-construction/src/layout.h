// ===========================================================================
// src/layout.h  --  Shared (host + device) pangenome 1-D layout primitives
// ---------------------------------------------------------------------------
// Project 3.30 : Pangenome Graph Construction
//
// WHAT THIS PROJECT COMPUTES  (the teaching-scoped slice of the catalog)
//   A pangenome graph (built by PGGB/ODGI from many genome assemblies) stores
//   the genomes as ONE sequence graph: NODES are sequence segments, and each
//   genome is a PATH that threads through a list of nodes. To *order* or
//   *visualize* the graph in 1-D, ODGI runs a "path-guided SGD layout"
//   (`odgi sort -p Ygs`, `odgi layout`): give every node a 1-D coordinate so
//   that nodes which sit close together ALONG A GENOME PATH also sit close in
//   the coordinate. That is exactly a graph-drawing STRESS problem.
//
//   We implement the deterministic teaching cousin of ODGI's stochastic layout:
//   STRESS MAJORIZATION (SMACOF) over a fixed set of node-pair "terms". A term
//   (i, j, d, w) says "nodes i and j are path-distance d apart (in base pairs),
//   with weight w; try to make their 1-D separation |x[i]-x[j]| equal d."
//   Minimizing the total weighted stress
//        E(x) = sum_terms  w * ( |x[i]-x[j]| - d )^2
//   spreads the nodes out so the drawing reflects genome co-linearity. SMACOF
//   minimises E by repeatedly jumping each node to the closed-form minimiser of a
//   quadratic upper bound (the Guttman transform, LO_term_numerator below) --
//   monotone, no learning rate, no divergence.
//
// WHY A GPU  (the catalog's "57.3x ODGI GPU layout" lesson, in miniature)
//   Real pangenomes have millions of nodes and BILLIONS of pairwise terms; the
//   layout is the dominant cost and it is a PARTICLE-PHYSICS-style simulation:
//   every term contributes independently to its two endpoints. That is
//   embarrassingly parallel over terms (one GPU thread per term) followed by a
//   SCATTER-REDUCTION of contributions into per-node accumulators. ODGI reports a
//   57.3x speed-up of its GPU layout over multi-core CPU for exactly this shape.
//
// THE DETERMINISM TRICK  (same idea as projects 5.01 and 11.09)
//   Many threads atomicAdd into the SAME node accumulator. A *float* atomicAdd is
//   order-dependent (floating-point addition is not associative), so the GPU sum
//   would vary run-to-run and would not match the CPU. We instead accumulate in
//   FIXED-POINT integers (atomicAdd on unsigned long long): integer adds COMMUTE,
//   so the reduction is order-independent => the GPU result is reproducible AND
//   equals the CPU result bit-for-bit. The quantisation is the *same* on both
//   sides (see LO_to_fixed below).
//
//   These helpers are __host__ __device__ (LO_HD) so the CPU reference and the
//   GPU kernels run byte-for-byte identical per-term math.  No CUDA-only types
//   appear here, so the plain host compiler can include this header too.
//
// READ THIS AFTER: nothing (start here), then reference_cpu.h -> kernels.cuh.
// ===========================================================================
#pragma once

#include <cstdint>   // std::int64_t, std::uint64_t

// LO_HD marks a function as callable from BOTH the host (CPU reference) and the
// device (GPU kernel). Under nvcc (__CUDACC__) it expands to the CUDA decorators;
// under the plain host compiler it expands to nothing.
#ifdef __CUDACC__
#define LO_HD __host__ __device__
#else
#define LO_HD
#endif

// ---------------------------------------------------------------------------
// A single layout "term": a soft constraint between two nodes.
//   i, j     : node indices (endpoints of the spring).
//   target_d : desired 1-D separation, in base pairs (the path distance).
//   weight   : how strongly to enforce it. ODGI weights a term by 1/d^2 so that
//              SHORT path distances (adjacent nodes) dominate -- they must be
//              placed precisely, while far-apart nodes may stretch. We store the
//              weight explicitly so the CPU and GPU read the identical number.
// This is a plain POD struct: it lives in a flat array, one element per term,
// and is read by exactly one thread on the GPU.
// ---------------------------------------------------------------------------
struct LayoutTerm {
    int    i;
    int    j;
    double target_d;
    double weight;
};

// ---------------------------------------------------------------------------
// WHY SMACOF (the Guttman transform) AND NOT GRADIENT STEPS
//   A naive "move each node a little down the stress gradient" needs a carefully
//   tuned step size: too big and the FULL-BATCH update (where a node feels the
//   sum of ALL its terms at once) overshoots and DIVERGES. Stress MAJORIZATION /
//   SMACOF avoids this entirely. It replaces the stress E(x) with a quadratic
//   upper bound that touches E at the current x, then jumps to that bound's exact
//   minimiser. That minimiser, for weighted 1-D MDS, is a closed form (the
//   "Guttman transform"):
//
//        x_i_new = ( sum_j w_ij * ( x_j + d_ij * sign(x_i - x_j) ) )
//                  -------------------------------------------------
//                  ( sum_j w_ij )
//
//   i.e. each node moves to the WEIGHTED AVERAGE of where each of its terms wants
//   it (a neighbour offset by the target distance, on the correct side). This is
//   UNCONDITIONALLY CONVERGENT -- stress never increases -- so there is no learning
//   rate to tune and no divergence. ODGI's path-SGD is the stochastic cousin of
//   this; we use the deterministic full-batch form so CPU and GPU match exactly.
//
// FIXED-POINT ACCUMULATION (the determinism trick)
//   Each sweep, every node accumulates two sums by atomicAdd: a NUMERATOR
//   (sum of w*(neighbour +/- d)) and a DENOMINATOR (sum of w). Float atomics are
//   order-dependent, so we quantise each contribution to an integer count of
//   "quanta" and accumulate those (integer adds commute -> deterministic and
//   CPU-matching). LO_SCALE sets the resolution: 2^20 ~ 1e6 quanta per unit gives
//   ~6 significant digits; signed 64-bit accumulators hold sums of millions of
//   coordinate-scale terms without overflow.
// ---------------------------------------------------------------------------
// constexpr (not plain `static const`): a compile-time constant is usable in
// BOTH host and device code. A namespace-scope `static const double` would have
// internal linkage with no device-side storage, so nvcc rejects it inside a
// __device__ function -- constexpr fixes that.
constexpr double LO_SCALE = 1048576.0;   // = 2^20 fixed-point quanta per unit

// Quantise a (signed) real to an integer count of quanta, rounding to nearest.
//   We branch on the sign so rounding is symmetric (round-half-away-from-zero) and
//   identical on host and device regardless of the libm implementation. Signed
//   because numerator contributions and positions can be negative.
LO_HD inline long long LO_to_fixed(double v) {
    if (v >= 0.0) return (long long)(v * LO_SCALE + 0.5);
    else          return (long long)(v * LO_SCALE - 0.5);
}

// Convert an accumulated integer count of quanta back to a real (double).
LO_HD inline double LO_from_fixed(long long fixed) {
    return (double)fixed / LO_SCALE;
}

// ---------------------------------------------------------------------------
// THE ONE TRUE PER-TERM CONTRIBUTION  (called by BOTH the CPU loop and the GPU)
//   For a single term (i, j, d, w) and the current positions xi, xj, compute the
//   Guttman-transform contribution this term makes to ENDPOINT `self` (which must
//   be i or j). `self_x` is that endpoint's current position; `other_x` is the
//   opposite endpoint's. We return the NUMERATOR contribution; the denominator
//   contribution is simply `w` (added by the caller).
//
//   sign(self_x - other_x) chooses which side of `other` the target wants `self`:
//   if self is currently to the right of other, the target keeps it to the right
//   (other + d); if to the left, (other - d). When the two coincide we break the
//   tie deterministically toward +1 so host and device agree.
//
//   Writing this once is the whole point of the shared header: the CPU reference
//   and the GPU kernel cannot drift apart in how they compute a term.
//
//   Parameters:
//     self_x  : current coordinate of the endpoint we are contributing to (bp).
//     other_x : current coordinate of the opposite endpoint (bp).
//     d       : the term's target separation (bp, > 0).
//     w       : the term's weight (1/d^2 in our construction).
//   Returns: w * ( other_x + d * sign(self_x - other_x) ), the numerator term.
// ---------------------------------------------------------------------------
LO_HD inline double LO_term_numerator(double self_x, double other_x,
                                      double d, double w) {
    const double diff = self_x - other_x;            // which side is `self` on?
    const double sign = (diff >= 0.0) ? 1.0 : -1.0;  // tie -> +1 (deterministic)
    // The target wants `self` a distance d from `other`, on `self`'s current side.
    return w * (other_x + d * sign);
}
