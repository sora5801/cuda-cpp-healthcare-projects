// ===========================================================================
// src/hic.h  --  Shared (host + device) Hi-C primitives: COO entry, fixed-point
//                row-sum quantization, and the ICE bias update formula.
// ---------------------------------------------------------------------------
// Project 3.15 : Hi-C / 3D Genome Contact Analysis  (see ../THEORY.md for "why")
//
// WHAT THIS PROJECT COMPUTES
//   A Hi-C experiment counts, for every pair of genomic bins (i,j), how often
//   the two loci were found cross-linked together -- i.e. how physically close
//   they are inside the nucleus. The result is a large, SYMMETRIC, SPARSE
//   contact matrix M (genome_bins x genome_bins). Raw counts are biased: some
//   bins are mappable/cut more often than others, inflating their whole row.
//
//   STEP 1 -- ICE BALANCING (Iterative Correction & Eigenvector decomposition,
//   Imakaev 2012). We seek a single per-bin BIAS vector b[] such that the
//   corrected matrix  M'_{ij} = M_{ij} / (b_i * b_j)  has every row summing to
//   the same constant (each bin is "equally visible"). ICE finds b[] by a fixed
//   point iteration: repeatedly (a) compute each row's current sum, (b) fold the
//   row sum into the bias, (c) rescale. This is the classic
//   matrix-balancing / Sinkhorn-Knopp idea. The hot loop is a SPARSE
//   MATRIX-VECTOR-like reduction (one row sum per bin) done O(20-50) times.
//
//   STEP 2 -- INSULATION SCORE + TAD BOUNDARIES. Topologically Associating
//   Domains (TADs) are square blocks along the diagonal of the balanced matrix
//   with many internal contacts and few across their borders. The INSULATION
//   SCORE of bin k is the mean balanced contact inside a small "diamond" window
//   straddling the diagonal at k; it dips sharply at a domain border (few cross-
//   border contacts). LOCAL MINIMA of the insulation score are called TAD
//   BOUNDARIES. This is exactly what tools like cooltools/hicexplorer compute.
//
// WHY A GPU
//   At 1 kb resolution a genome has ~3,000,000 bins; the contact matrix has up
//   to ~10^9 nonzeros. Each ICE iteration is a reduction over all nonzeros, and
//   we do dozens of them -- that is the bottleneck this project parallelises.
//   The reduction (each nonzero contributes to TWO row sums, i and j) is a
//   SCATTER: many threads add into the same row-sum bin -> atomicAdd. We make
//   that atomic add DETERMINISTIC (and exactly CPU-matching) with FIXED-POINT
//   integers (see HIC_SCALE below), because float atomics are non-associative.
//
//   The per-element math lives HERE as HIC_HD (= __host__ __device__) inline
//   functions so the CPU reference (reference_cpu.cpp, host compiler) and the
//   GPU kernels (kernels.cu, nvcc) run byte-for-byte identical arithmetic. This
//   is the "shared core" idiom from docs/PATTERNS.md §2.
//
// READ THIS BEFORE: reference_cpu.h, kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <cstdint>

// HIC_HD expands to __host__ __device__ only when this header is pulled through
// nvcc (which defines __CUDACC__). When the plain host compiler (cl.exe / g++)
// compiles reference_cpu.cpp, the decorators do not exist, so we erase them.
// This single trick lets ONE definition serve both the CPU and the GPU.
#ifdef __CUDACC__
#define HIC_HD __host__ __device__
#else
#define HIC_HD
#endif

// ---------------------------------------------------------------------------
// One stored contact: matrix entry (i, j) with raw count `count`.
//   We keep ONLY the upper triangle (i <= j) because Hi-C matrices are
//   symmetric -- storing both (i,j) and (j,i) would double the memory and the
//   work for no information gain. The diagonal (i == j) is kept once.
//   When we reduce row sums, an off-diagonal entry contributes to BOTH row i and
//   row j; a diagonal entry contributes to row i once. (See accumulate logic in
//   reference_cpu.cpp / kernels.cu -- both follow this same rule.)
// ---------------------------------------------------------------------------
struct CooEntry {
    int    i;       // row bin index   (0 <= i <= j < n_bins)
    int    j;       // column bin index
    double count;   // raw observed contact count (a non-negative integer in the
                    // sample, stored as double so the corrected value is exact)
};

// ---------------------------------------------------------------------------
// FIXED-POINT SCALE for the deterministic row-sum reduction.
//   Each ICE iteration sums corrected contributions M_{ij}/(b_i b_j) into a
//   per-bin accumulator. On the GPU many threads add into the same accumulator
//   concurrently; a FLOATING-POINT atomicAdd would sum them in a nondeterministic
//   order, and float addition is not associative -> the result would vary run to
//   run and would not match the serial CPU sum.
//
//   Fix: quantize each contribution to a 64-bit INTEGER number of "milliquanta"
//   (multiply by HIC_SCALE and round) and atomicAdd those integers. Integer
//   addition COMMUTES, so the integer sum is identical regardless of thread
//   order -- deterministic AND bit-identical to the CPU, which sums the same
//   quantized integers. We divide back by HIC_SCALE at the end.
//
//   Choice of scale: corrected contacts after the first normalisation are O(1),
//   a row has at most a few thousand nonzeros in the sample, and a uint64 holds
//   ~1.8e19. HIC_SCALE = 1e9 keeps ~9 decimal digits of each contribution while
//   leaving >1e9 headroom for the per-row sum -- comfortably safe here.
//
//   NOTE: `constexpr` (not `static const`) so the value is a compile-time
//   constant usable in BOTH host and __device__ code. A plain `static const`
//   double in a header gets host-only storage, and nvcc rejects reading it from
//   a __device__ function ("identifier undefined in device code").
// ---------------------------------------------------------------------------
constexpr double HIC_SCALE = 1.0e9;

// Quantize a non-negative corrected contribution to fixed-point integer quanta.
//   `+ 0.5` rounds to nearest (contributions are >= 0, so this is correct
//   rounding). Identical on host and device, so CPU and GPU quantize the same.
HIC_HD inline unsigned long long hic_to_fixed(double contribution) {
    return static_cast<unsigned long long>(contribution * HIC_SCALE + 0.5);
}

// Convert a fixed-point row-sum accumulator back to a floating-point row sum.
HIC_HD inline double hic_from_fixed(unsigned long long quanta) {
    return static_cast<double>(quanta) / HIC_SCALE;
}

// ---------------------------------------------------------------------------
// The ONE corrected-contact formula, shared by CPU and GPU.
//   Given raw count and the two endpoint biases, return the balanced contact
//   M'_{ij} = count / (b_i * b_j). Masked/empty bins carry bias 0; we guard
//   against divide-by-zero by returning 0 (a masked bin contributes nothing).
// ---------------------------------------------------------------------------
HIC_HD inline double hic_corrected(double count, double bias_i, double bias_j) {
    const double denom = bias_i * bias_j;
    return (denom > 0.0) ? (count / denom) : 0.0;
}
