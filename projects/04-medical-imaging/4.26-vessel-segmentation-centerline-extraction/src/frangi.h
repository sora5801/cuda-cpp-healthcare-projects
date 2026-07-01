// ===========================================================================
// src/frangi.h  --  Shared (host + device) Frangi vesselness core
// ---------------------------------------------------------------------------
// Project 4.26 : Vessel Segmentation & Centerline Extraction
//
// WHAT THIS FILE IS
//   The *per-voxel physics* of the Frangi vesselness filter, written ONCE as a
//   set of `__host__ __device__` inline functions. The CPU reference and the GPU
//   kernel both call these, so they run byte-for-byte identical math -- the key
//   idiom that makes CPU-vs-GPU verification exact (docs/PATTERNS.md section 2).
//
// THE IDEA (why a vesselness filter exists)
//   A blood vessel in a 3-D CT-angiography volume is a bright TUBE on a darker
//   background. Locally, a tube looks like a ridge: the image intensity barely
//   changes ALONG the vessel but drops off sharply ACROSS it in the two
//   perpendicular directions. The second-order local shape of the intensity is
//   captured by the HESSIAN matrix H (the 3x3 matrix of second derivatives).
//   Its three eigenvalues (lambda1, lambda2, lambda3, sorted by magnitude)
//   describe that shape:
//       |lambda1| ~ 0        (flat along the vessel axis)
//       |lambda2|, |lambda3| large and NEGATIVE (bright tube -> concave across)
//   Frangi et al. (1998) turned this into a single "vesselness" score in [0,1]
//   by combining three geometric ratios (R_A, R_B, S). This file computes, per
//   voxel: the Hessian (by finite differences), its eigenvalues (closed-form
//   symmetric 3x3), and the Frangi score.
//
// WHY CLOSED-FORM EIGENVALUES (not iterative Jacobi)
//   The catalog mentions "Jacobi iteration". We instead use the ANALYTIC
//   (Cardano) eigenvalue formula for a symmetric 3x3 matrix. Two reasons:
//     (1) It is a fixed, branch-light sequence of arithmetic -> the CPU and GPU
//         produce the SAME result to ~1e-9 (an iterative solver's variable
//         iteration count would make exact parity fragile). See THEORY section 5.
//     (2) It teaches the beautiful fact that a symmetric 3x3 eigenproblem has a
//         known trigonometric solution -- no iteration needed.
//   Jacobi is discussed in THEORY as the production alternative (it also yields
//   eigenVECTORS, which we do not need for the scalar vesselness score).
//
// NOTATION: FR_HD expands to __host__ __device__ under nvcc, to nothing on the
//   host compiler. Keep this header free of CUDA-only types (no __global__) so
//   the plain C++ compiler can include it for reference_cpu.cpp.
//
// READ THIS AFTER: reference_cpu.h (the Volume struct).  READ BEFORE: kernels.cu.
// ===========================================================================
#pragma once

#include <cstddef>   // std::size_t
#include <cmath>     // std::sqrt, std::acos, std::fabs, std::fmin/fmax

#ifdef __CUDACC__
#define FR_HD __host__ __device__
#else
#define FR_HD
#endif

// ---------------------------------------------------------------------------
// vox_idx: flat (row-major) index of voxel (x,y,z) in a nx*ny*nz volume.
//   Layout: x fastest, then y, then z. Consecutive x are contiguous in memory,
//   so a warp of threads walking x reads a coalesced cache line on the GPU.
// ---------------------------------------------------------------------------
FR_HD inline std::size_t vox_idx(int x, int y, int z, int nx, int ny) {
    return (static_cast<std::size_t>(z) * ny + y) * nx + x;
}

// ---------------------------------------------------------------------------
// FrangiParams: the tunable knobs of the Frangi filter (all dimensionless here;
//   intensities are treated in raw units). Defaults follow Frangi 1998.
//   * alpha : sensitivity to R_A = |l2|/|l3| (distinguishes plate vs. line).
//   * beta  : sensitivity to R_B = |l1|/sqrt(|l2 l3|) (distinguishes blob vs. line).
//   * c     : sensitivity to S = ||H||_F (the "structureness"; suppresses noise
//             in flat regions where all eigenvalues are tiny).
//   * sigma : Gaussian pre-smoothing scale in voxels. The Hessian is computed on
//             the smoothed image so the filter responds to vessels of ~that
//             radius. Single-scale here; multi-scale is an exercise (THEORY).
//   * bright_vessels : true if vessels are BRIGHTER than background (CTA); this
//             flips the eigenvalue-sign gate. CTA contrast -> true.
// ---------------------------------------------------------------------------
struct FrangiParams {
    double alpha = 0.5;
    double beta  = 0.5;
    double c     = 500.0;   // scaled to the synthetic intensity range (see data/README)
    double sigma = 1.0;
    int    bright_vessels = 1;
};

// ---------------------------------------------------------------------------
// A tiny helper: sort three doubles by ABSOLUTE value, ascending, so that on
//   return |a| <= |b| <= |c|. Frangi's ratios assume this ordering
//   (lambda1 = smallest magnitude, lambda3 = largest magnitude).
//   Written as a fixed sequence of compare-swaps (a sorting network) -> no data-
//   dependent loop, identical on CPU and GPU.
// ---------------------------------------------------------------------------
FR_HD inline void sort_abs3(double& a, double& b, double& c) {
    // compare-swap by |.|; swap the actual (signed) values so signs are kept.
    if (std::fabs(a) > std::fabs(b)) { double t = a; a = b; b = t; }
    if (std::fabs(b) > std::fabs(c)) { double t = b; b = c; c = t; }
    if (std::fabs(a) > std::fabs(b)) { double t = a; a = b; b = t; }
}

// ---------------------------------------------------------------------------
// eig_sym3: eigenvalues of a real SYMMETRIC 3x3 matrix in closed form.
//
//   Input: the 6 unique entries of
//         [ h00 h01 h02 ]
//     H = [ h01 h11 h12 ]
//         [ h02 h12 h22 ]
//   Output: e0 <= e1 <= e2  (the three real eigenvalues, ascending by VALUE).
//
//   METHOD (Smith 1961 / the standard trigonometric formula). A symmetric matrix
//   has three real eigenvalues; they are the roots of the characteristic cubic
//   det(H - lambda I) = 0. For a symmetric matrix this cubic always has three
//   real roots, so Cardano's formula reduces to the "trigonometric" (casus
//   irreducibilis) branch:
//       q     = trace(H)/3                       (mean eigenvalue)
//       H'    = H - q I                          (shift so it is traceless)
//       p     = sqrt( sum(H'_ij^2) / 6 )         (a scale)
//       B     = H' / p                           (normalized, det in [-1,1])
//       phi   = acos( det(B)/2 ) / 3             (the key angle)
//       roots = q + 2 p cos( phi + k*2pi/3 ),  k=0,1,2
//   We clamp det(B)/2 to [-1,1] before acos to be robust to round-off, and we
//   special-case a diagonal matrix (p==0) to avoid a divide-by-zero.
//
//   This is ~40 flops per voxel -- trivial for the GPU, and (crucially) the same
//   flops on both sides, so results agree to ~1e-9 (THEORY section 5).
// ---------------------------------------------------------------------------
FR_HD inline void eig_sym3(double h00, double h01, double h02,
                           double h11, double h12, double h22,
                           double& e0, double& e1, double& e2) {
    const double q = (h00 + h11 + h22) / 3.0;         // mean of the diagonal
    // Deviatoric (traceless) part H' = H - q I; only the diagonal changes.
    const double a00 = h00 - q, a11 = h11 - q, a22 = h22 - q;
    // p2 = sum of squares of ALL entries of H' (off-diagonals counted twice).
    const double p2 = a00*a00 + a11*a11 + a22*a22
                    + 2.0 * (h01*h01 + h02*h02 + h12*h12);
    if (p2 <= 0.0) {
        // H' is the zero matrix -> H was q*I -> all three eigenvalues equal q.
        e0 = e1 = e2 = q;
        return;
    }
    const double p = std::sqrt(p2 / 6.0);
    const double ip = 1.0 / p;
    // B = H'/p. Compute det(B) = det(H')/p^3.
    const double b00 = a00*ip, b11 = a11*ip, b22 = a22*ip;
    const double b01 = h01*ip, b02 = h02*ip, b12 = h12*ip;
    // 3x3 determinant of the symmetric B (expanded along the first row).
    double detB = b00*(b11*b22 - b12*b12)
                - b01*(b01*b22 - b12*b02)
                + b02*(b01*b12 - b11*b02);
    double r = detB / 2.0;
    // Clamp for acos: round-off can push r a hair outside [-1,1].
    if (r < -1.0) r = -1.0; else if (r > 1.0) r = 1.0;
    const double phi = std::acos(r) / 3.0;
    const double PI2_3 = 2.0943951023931953;  // 2*pi/3
    // The three roots, already ascending because cos is decreasing on [0,pi].
    e2 = q + 2.0 * p * std::cos(phi);              // largest
    e0 = q + 2.0 * p * std::cos(phi + 2.0*PI2_3);  // smallest
    e1 = 3.0 * q - e0 - e2;                        // middle (trace is invariant)
}

// ---------------------------------------------------------------------------
// frangi_response: the Frangi vesselness score from three eigenvalues.
//   Inputs l1,l2,l3 are the eigenvalues SORTED BY MAGNITUDE (|l1|<=|l2|<=|l3|).
//   Returns a score in [0,1]; higher = more tube-like.
//
//   The three geometric ratios (Frangi 1998, eq. 11-13):
//     R_A = |l2| / |l3|         -> ~1 for a line, ~0 for a plate/sheet
//     R_B = |l1| / sqrt(|l2*l3|)-> ~0 for a line, ~1 for a blob
//     S   = sqrt(l1^2+l2^2+l3^2)-> the Frobenius norm ("structureness"); small
//                                  in flat noisy regions, large on real edges.
//   Combined:
//     V = (1 - exp(-R_A^2/2a^2)) * exp(-R_B^2/2b^2) * (1 - exp(-S^2/2c^2))
//   For BRIGHT vessels on a dark background (CTA), a valid tube requires the two
//   large eigenvalues to be NEGATIVE (image is concave across the vessel); if
//   they are not, the score is forced to 0. (Flip for dark-on-bright.)
// ---------------------------------------------------------------------------
FR_HD inline double frangi_response(double l1, double l2, double l3,
                                    const FrangiParams& fp) {
    // Sign gate: bright tube => l2,l3 < 0. Reject the wrong polarity.
    if (fp.bright_vessels) {
        if (l2 > 0.0 || l3 > 0.0) return 0.0;
    } else {
        if (l2 < 0.0 || l3 < 0.0) return 0.0;
    }
    const double abs1 = std::fabs(l1), abs2 = std::fabs(l2), abs3 = std::fabs(l3);
    // Guard degenerate denominators (a truly flat voxel): no structure -> 0.
    if (abs3 < 1e-12) return 0.0;

    const double Ra = abs2 / abs3;                 // plate vs. line
    const double Rb = abs1 / std::sqrt(abs2 * abs3 + 1e-300);  // blob vs. line
    const double S  = std::sqrt(l1*l1 + l2*l2 + l3*l3);        // structureness

    const double a2 = 2.0 * fp.alpha * fp.alpha;
    const double b2 = 2.0 * fp.beta  * fp.beta;
    const double c2 = 2.0 * fp.c     * fp.c;

    const double vA = 1.0 - std::exp(-(Ra*Ra) / a2);   // ->1 as Ra->1 (line)
    const double vB = std::exp(-(Rb*Rb) / b2);         // ->1 as Rb->0 (line)
    const double vS = 1.0 - std::exp(-(S*S) / c2);     // ->1 on real structure
    return vA * vB * vS;
}
