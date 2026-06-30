// ===========================================================================
// src/dcc_core.h  --  The ONE shared per-pair physics, used by CPU and GPU
// ---------------------------------------------------------------------------
// Project 2.17 : Allosteric Network Analysis
//
// WHY THIS FILE EXISTS  (PATTERNS.md section 2: the shared __host__ __device__ core)
//   The heavy numerical kernel of this project is the Dynamical Cross-Correlation
//   matrix (DCC): for every ordered pair of residues (i, j) we average, over all
//   T trajectory frames, the dot product of their displacement-from-mean vectors,
//   then normalize. If the CPU reference and the GPU kernel implemented that
//   average with even slightly different code, their floating-point round-off
//   would diverge and "verify GPU == CPU" would become a fuzzy guess.
//
//   The fix is to write the per-pair formula EXACTLY ONCE, here, as an inline
//   function tagged `__host__ __device__`. The CPU reference (reference_cpu.cpp,
//   compiled by cl.exe) calls it; the GPU kernel (kernels.cu, compiled by nvcc)
//   calls the very same source. Same operations, same order  ->  the two results
//   agree to the last bit and our verification can use an EXACT tolerance.
//
//   KEEP THIS HEADER CUDA-LIGHT. It must compile under the plain host compiler,
//   so it may use the HD macro but must NOT contain `__global__`, kernel launches,
//   <<<>>> syntax, or any CUDA runtime type. Only <cmath> and plain C++.
//
// READ THIS BEFORE: reference_cpu.cpp and kernels.cu (both include this file).
// ===========================================================================
#pragma once

#include <cmath>     // std::sqrt, std::fabs, std::log
#include <cstddef>   // std::size_t

// ---------------------------------------------------------------------------
// HD : "host + device" decorator.
//   When this header is fed to nvcc (because kernels.cu includes it), __CUDACC__
//   is defined and we expand HD to `__host__ __device__`, so the SAME function
//   can be called from CPU code AND from inside a kernel. When the plain host
//   compiler builds reference_cpu.cpp, __CUDACC__ is NOT defined, so HD expands
//   to nothing and the function is an ordinary inline host function.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ---------------------------------------------------------------------------
// Trajectory layout, defined once so every file agrees on the indexing.
//
//   We store the trajectory as a flat array of single-precision floats in
//   FRAME-MAJOR, then RESIDUE-MAJOR, then XYZ order:
//
//       coords[ (t * N + i) * 3 + c ]
//
//   where  t in [0,T)  is the frame,  i in [0,N)  is the residue, and
//   c in {0,1,2} is the x/y/z component. This row-major packing means all three
//   coordinates of one residue in one frame are contiguous (good for the per-pair
//   dot product), and consecutive residues within a frame are contiguous too.
//
//   The displacement-from-mean is  d_i(t) = r_i(t) - <r_i>, where <r_i> is the
//   residue's average position over all frames (its "equilibrium" point). We
//   precompute the mean per residue (mean[i*3 + c]) so both the CPU and GPU work
//   from identical, already-centered numbers.
// ---------------------------------------------------------------------------

// Index of component c of residue i in frame t, inside the flat coords array.
HD inline std::size_t coord_index(int t, int i, int c, int N) {
    return (static_cast<std::size_t>(t) * N + i) * 3 + c;
}

// ---------------------------------------------------------------------------
// dcc_pair : the single source of truth for one DCC matrix entry C[i][j].
//
//   THE MATH (THEORY.md section "The math"):
//
//       Cov(i,j) = (1/T) * sum_t  d_i(t) . d_j(t)
//       C[i][j]  = Cov(i,j) / sqrt( Cov(i,i) * Cov(j,j) )
//
//   The numerator is the time-averaged dot product of the two residues'
//   displacement vectors: it is large and positive when i and j move in the
//   same direction together (correlated motion), large and negative when they
//   move oppositely (anti-correlated), and ~0 when their motions are unrelated.
//   Dividing by the geometric mean of their self-covariances (the variances of
//   their own motion) rescales this to the familiar Pearson range [-1, +1], so
//   the result does not just reflect "which residue wiggles more".
//
//   We pass the precomputed per-residue means in `mean` so this function does
//   not re-derive them (and so CPU and GPU center the data identically). The
//   accumulation is done in `double` even though the trajectory is `float`: the
//   sum over T frames of products can lose precision badly in float, and double
//   accumulation is the cheap, standard guard. Both CPU and GPU use double here,
//   which is exactly why they agree to the bit.
//
//   PARAMETERS
//     coords : [T*N*3] flat trajectory (float), frame-major (see coord_index)
//     mean   : [N*3]   per-residue average position (double), precomputed once
//     i, j   : the two residue indices (0-based) whose correlation we want
//     T, N   : number of frames and number of residues
//   RETURNS
//     C[i][j] in [-1, +1]; exactly +1 on the diagonal (i == j) by construction.
//
//   COMPLEXITY: O(T) per call. The full matrix is N*N calls -> O(N^2 * T),
//   which is the bottleneck this project parallelizes on the GPU (one thread per
//   (i,j) entry; see kernels.cu).
// ---------------------------------------------------------------------------
HD inline double dcc_pair(const float* coords, const double* mean,
                          int i, int j, int T, int N) {
    double cov_ij = 0.0;   // sum_t d_i(t).d_j(t)
    double cov_ii = 0.0;   // sum_t d_i(t).d_i(t)  (variance of residue i's motion)
    double cov_jj = 0.0;   // sum_t d_j(t).d_j(t)  (variance of residue j's motion)

    // Walk every frame, accumulating the three covariance sums in lock-step so
    // the normalization uses the SAME frames as the numerator.
    for (int t = 0; t < T; ++t) {
        // Displacement-from-mean vectors of residues i and j in this frame.
        const double dix = coords[coord_index(t, i, 0, N)] - mean[i * 3 + 0];
        const double diy = coords[coord_index(t, i, 1, N)] - mean[i * 3 + 1];
        const double diz = coords[coord_index(t, i, 2, N)] - mean[i * 3 + 2];
        const double djx = coords[coord_index(t, j, 0, N)] - mean[j * 3 + 0];
        const double djy = coords[coord_index(t, j, 1, N)] - mean[j * 3 + 1];
        const double djz = coords[coord_index(t, j, 2, N)] - mean[j * 3 + 2];

        cov_ij += dix * djx + diy * djy + diz * djz;  // 3-D dot product, summed over t
        cov_ii += dix * dix + diy * diy + diz * diz;
        cov_jj += djx * djx + djy * djy + djz * djz;
    }

    // Normalize to the Pearson correlation coefficient. The 1/T factors in the
    // numerator and denominator cancel, so we never even form them. Guard the
    // pathological case of a residue that never moves (cov == 0) to avoid 0/0.
    const double denom = std::sqrt(cov_ii * cov_jj);
    if (denom <= 0.0) return (i == j) ? 1.0 : 0.0;
    return cov_ij / denom;
}

// ---------------------------------------------------------------------------
// comm_weight : turn a correlation magnitude into a communication "distance".
//
//   Allostery is about INFORMATION FLOW through the residue network, not just
//   raw correlation. The standard trick (Bio3D, WORDOM, Sethi et al. 2009) is to
//   convert each strong correlation into a short edge:
//
//       w_ij = -log( |C[i][j]| )
//
//   |C| near 1  ->  w near 0  (a strong, "cheap-to-traverse" coupling);
//   |C| near 0  ->  w huge    (a weak, "expensive" link). The shortest path in
//   this weighted graph is then the route of strongest end-to-end correlation:
//   the candidate allosteric communication pathway. We only build edges between
//   residues that are in spatial contact (an edge cannot exist where atoms are
//   not near each other), enforced by the caller, so a signal must physically
//   hop residue-to-residue rather than teleport across the protein.
//
//   This same formula is used on BOTH sides so the network the GPU path and the
//   CPU path analyze is byte-identical.
// ---------------------------------------------------------------------------
HD inline double comm_weight(double corr) {
    double a = std::fabs(corr);
    // Clamp |C| into (eps, 1] so the logarithm is finite and non-positive's edge
    // never becomes negative. eps = 1e-6 means "essentially uncorrelated".
    const double eps = 1.0e-6;
    if (a < eps) a = eps;
    if (a > 1.0) a = 1.0;
    return -std::log(a);   // >= 0 always; 0 only for perfect correlation
}
