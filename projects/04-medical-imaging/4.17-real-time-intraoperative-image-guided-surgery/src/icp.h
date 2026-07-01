// ===========================================================================
// src/icp.h  --  Shared (host + device) ICP primitives  (the "one true math")
// ---------------------------------------------------------------------------
// Project 4.17 : Real-Time Intraoperative / Image-Guided Surgery
//
// WHAT THIS PROJECT COMPUTES
//   RIGID point-cloud registration by Iterative Closest Point (ICP). In
//   image-guided surgery (IGS) we have a PRE-OPERATIVE surface (say, a tumour
//   or organ surface extracted from the planning MRI/CT) and an INTRA-OPERATIVE
//   surface (points digitized during surgery -- e.g. from a tracked pointer,
//   ultrasound, or a depth camera). To overlay the plan onto the patient we
//   must find the rigid motion (rotation R + translation t) that best maps the
//   moving cloud P onto the fixed cloud Q. That map is what lets the navigation
//   system draw the plan in the right place on the live view.
//
//   ICP alternates two steps until the fit stops improving:
//     (1) CORRESPOND : for each moving point p_i, find its NEAREST fixed point
//                      q_{c(i)} (a brute-force nearest-neighbour search here).
//                      -> embarrassingly parallel: one GPU thread per moving pt.
//     (2) ALIGN      : given those pairs, solve for the rigid (R,t) that best
//                      maps {p_i} onto {q_{c(i)}} in the least-squares sense.
//                      This is the classic Kabsch/Arun/Horn solution:
//                        - subtract centroids of both matched sets,
//                        - form the 3x3 cross-covariance  H = sum (p'_i)(q'_i)^T,
//                        - SVD  H = U S V^T,  then  R = V * diag(1,1,det) * U^T,
//                        - t = centroid(Q) - R * centroid(P).
//                      H is a REDUCTION over all points -> the second parallel
//                      pattern (accumulate 9 covariance entries + 2 centroids).
//   Apply (R,t) to P, and repeat. Each iteration lowers the RMS pairing error.
//
// WHY A GPU  (see the catalog deep-dive & ../THEORY.md)
//   Intra-operative registration has a hard latency budget (< a few seconds).
//   Real surfaces have 10^4-10^6 points, and the CORRESPOND step is O(|P|*|Q|)
//   with brute force -- the dominant cost. Every moving point's search is
//   independent, so it maps perfectly onto GPU threads. The covariance for the
//   ALIGN step is then a parallel REDUCTION. The tiny 3x3 SVD that follows is
//   done ONCE per iteration on the host (negligible, and easier to keep exact).
//
// DETERMINISM TRICK  (identical idea to project 11.09 flow-cytometry k-means)
//   The reduction that builds H and the two centroids is a sum over many points.
//   A floating-point atomicAdd is order-dependent (non-associative) -> the GPU
//   sum would vary run to run and drift from the CPU. So we accumulate in
//   FIXED-POINT integers (atomicAdd on signed 64-bit): integer adds commute, so
//   the GPU reduction is reproducible AND equals the CPU reduction exactly. The
//   host then divides the integer sums back to doubles and runs the SAME 3x3
//   SVD on both sides -> CPU and GPU produce byte-identical transforms.
//
//   The per-point helpers below are __host__ __device__ (ICP_HD) so the CPU
//   reference (reference_cpu.cpp) and the GPU kernels (kernels.cu) share ONE
//   copy of the math. No CUDA-only types appear here, so the host compiler can
//   include it too. (HD-macro idiom -- docs/PATTERNS.md section 2.)
//
// READ THIS BEFORE: reference_cpu.cpp, kernels.cu, main.cu.
// ===========================================================================
#pragma once

#include <cstdint>   // int64_t
#include <cmath>     // std::sqrt, std::fabs

// The HD macro: when compiled by nvcc (__CUDACC__ defined) the helpers become
// callable from BOTH host and device; under the plain host compiler the
// decorators simply vanish. This is what gives us CPU/GPU parity for free.
#ifdef __CUDACC__
#define ICP_HD __host__ __device__
#else
#define ICP_HD
#endif

// ---------------------------------------------------------------------------
// A 3-D point. Plain-old-data so it copies trivially between host and device.
// Coordinates are in millimetres (mm) -- a natural unit for surgical space.
// ---------------------------------------------------------------------------
struct Vec3 {
    float x, y, z;
};

// ---------------------------------------------------------------------------
// A rigid transform: rotate by the 3x3 matrix R (row-major, R[row][col]) then
// translate by t.  y = R * x + t.  We store R as doubles because it is built
// from an SVD and reused across iterations -- keeping it double avoids slow
// error accumulation over the ~dozen iterations of ICP.
// ---------------------------------------------------------------------------
struct Rigid {
    double R[3][3];   // rotation (orthonormal, det = +1)
    double t[3];      // translation (mm)
};

// The identity transform (R = I, t = 0): ICP's starting guess.
ICP_HD inline Rigid rigid_identity() {
    Rigid g;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) g.R[r][c] = (r == c) ? 1.0 : 0.0;
        g.t[r] = 0.0;
    }
    return g;
}

// Apply a rigid transform to a point:  out = R * p + t.  Used by BOTH the CPU
// reference and the GPU kernel to move the source cloud each iteration, so the
// moved coordinates are identical on both sides.
ICP_HD inline Vec3 rigid_apply(const Rigid& g, const Vec3& p) {
    Vec3 out;
    out.x = (float)(g.R[0][0] * p.x + g.R[0][1] * p.y + g.R[0][2] * p.z + g.t[0]);
    out.y = (float)(g.R[1][0] * p.x + g.R[1][1] * p.y + g.R[1][2] * p.z + g.t[1]);
    out.z = (float)(g.R[2][0] * p.x + g.R[2][1] * p.y + g.R[2][2] * p.z + g.t[2]);
    return out;
}

// Squared Euclidean distance between two points (mm^2). We compare SQUARED
// distances during the nearest-neighbour search because it avoids a sqrt in the
// inner loop and the ordering is identical (sqrt is monotincreasing).
ICP_HD inline double sqdist(const Vec3& a, const Vec3& b) {
    const double dx = (double)a.x - (double)b.x;
    const double dy = (double)a.y - (double)b.y;
    const double dz = (double)a.z - (double)b.z;
    return dx * dx + dy * dy + dz * dz;
}

// ---------------------------------------------------------------------------
// nearest_index: brute-force nearest-neighbour of point `p` among the `nq`
// fixed points `q`. Returns the index of the closest q (ties -> lowest index,
// via strict `<`, so the result is deterministic). This is the O(nq) heart of
// the CORRESPOND step; one call per moving point.
//
//   Production ICP replaces this linear scan with a k-d tree / octree
//   (O(log nq) per query). We keep brute force because it is the clearest
//   version to parallelise and to verify -- see ../THEORY.md "real world".
// ---------------------------------------------------------------------------
ICP_HD inline int nearest_index(const Vec3& p, const Vec3* q, int nq) {
    int best = 0;
    double best_d = sqdist(p, q[0]);
    for (int j = 1; j < nq; ++j) {
        const double d = sqdist(p, q[j]);
        if (d < best_d) { best_d = d; best = j; }
    }
    return best;
}

// ===========================================================================
// FIXED-POINT REDUCTION SUPPORT  (the determinism trick)
// ---------------------------------------------------------------------------
// The ALIGN step needs three sums over all corresponded points:
//   * sum of moving points        (to get centroid of P)
//   * sum of matched fixed points (to get centroid of Q)
//   * the 3x3 cross-covariance    H = sum (p_i - mean_P)(q_i - mean_Q)^T
// We accumulate all of these in INTEGER fixed-point so the parallel atomicAdd
// reduction on the GPU is order-independent and matches the CPU bit-for-bit.
//
// Scale choice: our synthetic surgical coordinates live in a box of a few
// hundred mm. Covariance entries are (mm)*(mm) ~ up to ~1e5, summed over up to
// ~1e5 points -> ~1e10 in magnitude. With COV_SCALE = 2^16 (~6.5e4) a single
// term is < ~1e10, and the running sum stays far under int64's ~9.2e18 ceiling.
// 2^16 keeps ~4-5 fractional digits of a millimetre -- far finer than any real
// surgical tracker (~0.1 mm), so quantization is physically negligible and the
// recovered transform is essentially exact (verified in main.cu).
// ===========================================================================
// NOTE: these MUST be `constexpr`, not `static const`. A plain `static const`
// object has host-only storage, so nvcc rejects reading it from device code
// ("identifier undefined in device code"). `constexpr` makes the value a
// compile-time constant that is legal in both host and device code -- exactly
// what we need since the kernel and the CPU reference both read these scales.
constexpr double COV_SCALE = 65536.0;         // 2^16 fixed-point scale for H terms
constexpr double POS_SCALE = 65536.0;         // 2^16 fixed-point scale for centroids

// Quantize a coordinate value (mm) to fixed-point integer. std::llround gives
// deterministic round-half-away-from-zero, identical on host and device.
ICP_HD inline int64_t to_fixed(double v, double scale) {
    // Manual round-half-away-from-zero (llround is not always a device builtin):
    return (int64_t)(v >= 0.0 ? (v * scale + 0.5) : (v * scale - 0.5));
}

// Convert an accumulated fixed-point sum back to a double value.
ICP_HD inline double from_fixed(int64_t s, double scale) {
    return (double)s / scale;
}

// ---------------------------------------------------------------------------
// AccumFixed: the 11 integer accumulators the ALIGN reduction fills, for ONE
// ICP iteration:  sum of P (3), sum of matched Q (3), cross terms of the
// covariance built from the CENTROID-RELATIVE points... but we cannot subtract
// the (unknown-until-summed) centroid inside the reduction. Standard trick:
// accumulate the RAW cross-products  sum p_i q_i^T  (9 entries) and the two raw
// sums, then form the centred covariance afterwards via the identity
//   sum (p_i-mp)(q_i-mq)^T = sum p_i q_i^T  -  n * mp * mq^T.
// So we need: sumP[3], sumQ[3], sumPQ[9], and the count n. All integer.
// ---------------------------------------------------------------------------
struct AccumFixed {
    int64_t sumP[3];    // sum of moving points p_i         (POS_SCALE fixed-pt)
    int64_t sumQ[3];    // sum of matched fixed points q_i  (POS_SCALE fixed-pt)
    int64_t sumPQ[9];   // sum of outer products p_i q_i^T  (COV_SCALE fixed-pt),
                        //   row-major: sumPQ[r*3 + c] accumulates p_r * q_c
    int64_t count;      // number of corresponded pairs n
};

// Zero every accumulator (host-side reset before each iteration's reduction).
ICP_HD inline void accum_zero(AccumFixed& a) {
    for (int i = 0; i < 3; ++i) { a.sumP[i] = 0; a.sumQ[i] = 0; }
    for (int i = 0; i < 9; ++i) a.sumPQ[i] = 0;
    a.count = 0;
}

// Add one corresponded pair (p, q) into the accumulators, in fixed-point. Used
// by the CPU reference directly and mirrored by the GPU kernel via atomicAdd.
// The outer product p*q^T is scaled by COV_SCALE (product of two mm values, so
// we scale ONCE, not twice, to keep the magnitude sane -- see COV_SCALE note).
ICP_HD inline void accum_pair(AccumFixed& a, const Vec3& p, const Vec3& q) {
    const double pv[3] = { (double)p.x, (double)p.y, (double)p.z };
    const double qv[3] = { (double)q.x, (double)q.y, (double)q.z };
    for (int r = 0; r < 3; ++r) {
        a.sumP[r] += to_fixed(pv[r], POS_SCALE);
        a.sumQ[r] += to_fixed(qv[r], POS_SCALE);
        for (int c = 0; c < 3; ++c)
            a.sumPQ[r * 3 + c] += to_fixed(pv[r] * qv[c], COV_SCALE);
    }
    a.count += 1;
}

// ===========================================================================
// 3x3 SYMMETRIC/GENERAL SVD via one-sided Jacobi  (host + device capable, but
// we only ever call it on the HOST -- once per iteration, on the tiny 3x3 H).
// ---------------------------------------------------------------------------
// Given a 3x3 matrix H, produce H = U * S * V^T with U,V orthonormal and S >= 0.
// We use the classic one-sided Jacobi SVD: repeatedly rotate pairs of columns
// of A (=H) to make them orthogonal; the accumulated column rotations form V,
// the normalized columns give U and the norms give the singular values S. It is
// tiny (3x3), converges in a handful of sweeps, and -- crucially -- runs the
// SAME code path on host for both the CPU reference and the GPU wrapper, so the
// recovered rotation is identical. (For big matrices you would call cuSOLVER;
// here a 3x3 hand-rolled solver is clearer and keeps CPU==GPU exact.)
// ===========================================================================

// A minimal 3x3 double matrix helper set (row-major m[r][c]).
struct Mat3 { double m[3][3]; };

ICP_HD inline Mat3 mat3_zero() { Mat3 A; for (int r=0;r<3;++r) for(int c=0;c<3;++c) A.m[r][c]=0.0; return A; }
ICP_HD inline Mat3 mat3_identity() { Mat3 A=mat3_zero(); A.m[0][0]=A.m[1][1]=A.m[2][2]=1.0; return A; }

// Matrix product C = A * B.
ICP_HD inline Mat3 mat3_mul(const Mat3& A, const Mat3& B) {
    Mat3 C = mat3_zero();
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) s += A.m[r][k] * B.m[k][c];
            C.m[r][c] = s;
        }
    return C;
}

// Transpose.
ICP_HD inline Mat3 mat3_transpose(const Mat3& A) {
    Mat3 T;
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) T.m[r][c] = A.m[c][r];
    return T;
}

// Determinant (used to guard against a reflection -> ensure det(R) = +1).
ICP_HD inline double mat3_det(const Mat3& A) {
    return A.m[0][0]*(A.m[1][1]*A.m[2][2]-A.m[1][2]*A.m[2][1])
         - A.m[0][1]*(A.m[1][0]*A.m[2][2]-A.m[1][2]*A.m[2][0])
         + A.m[0][2]*(A.m[1][0]*A.m[2][1]-A.m[1][1]*A.m[2][0]);
}

// One-sided Jacobi SVD of a 3x3 matrix A: fills U, S (as a diagonal length-3
// array), V so that A = U * diag(S) * V^T. Deterministic: fixed sweep count and
// fixed column order, so host and device agree exactly.
ICP_HD inline void svd3x3(const Mat3& A_in, Mat3& U, double S[3], Mat3& V) {
    // Work on a copy of A whose COLUMNS we orthogonalize in place. `Acol[k]` is
    // conceptually the k-th column; we keep A as a full matrix and index cols.
    Mat3 A = A_in;
    V = mat3_identity();               // accumulates the column rotations -> V

    // A "sweep" tries all 3 unordered column pairs (0,1),(0,2),(1,2). 30 sweeps
    // is far more than a 3x3 needs (it converges in ~5-8); the extra sweeps are
    // cheap and make convergence independent of the input -> deterministic.
    for (int sweep = 0; sweep < 30; ++sweep) {
        for (int p = 0; p < 3; ++p) {
            for (int q = p + 1; q < 3; ++q) {
                // Column dot products alpha=<a_p,a_p>, beta=<a_q,a_q>, gamma=<a_p,a_q>.
                double alpha = 0.0, beta = 0.0, gamma = 0.0;
                for (int r = 0; r < 3; ++r) {
                    alpha += A.m[r][p] * A.m[r][p];
                    beta  += A.m[r][q] * A.m[r][q];
                    gamma += A.m[r][p] * A.m[r][q];
                }
                if (gamma == 0.0) continue;                 // already orthogonal
                // Jacobi rotation angle that zeroes the (p,q) column overlap.
                const double zeta = (beta - alpha) / (2.0 * gamma);
                const double sign = (zeta >= 0.0) ? 1.0 : -1.0;
                const double t = sign / (std::fabs(zeta) + std::sqrt(1.0 + zeta * zeta));
                const double cc = 1.0 / std::sqrt(1.0 + t * t);   // cos
                const double ss = cc * t;                          // sin
                // Rotate columns p and q of A, and of V (to build the product).
                for (int r = 0; r < 3; ++r) {
                    const double ap = A.m[r][p], aq = A.m[r][q];
                    A.m[r][p] = cc * ap - ss * aq;
                    A.m[r][q] = ss * ap + cc * aq;
                    const double vp = V.m[r][p], vq = V.m[r][q];
                    V.m[r][p] = cc * vp - ss * vq;
                    V.m[r][q] = ss * vp + cc * vq;
                }
            }
        }
    }

    // Now A's columns are orthogonal; their norms are the singular values and
    // the normalized columns are U's columns.
    U = mat3_identity();
    for (int c = 0; c < 3; ++c) {
        double norm = 0.0;
        for (int r = 0; r < 3; ++r) norm += A.m[r][c] * A.m[r][c];
        norm = std::sqrt(norm);
        S[c] = norm;
        if (norm > 1e-300) {
            for (int r = 0; r < 3; ++r) U.m[r][c] = A.m[r][c] / norm;
        } else {
            // Degenerate column: keep the identity column (rank-deficient H).
            for (int r = 0; r < 3; ++r) U.m[r][c] = (r == c) ? 1.0 : 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// solve_rigid: the ALIGN step, given the finished fixed-point accumulators.
//   1. centroids  mp = sumP/n,  mq = sumQ/n.
//   2. centred covariance  H = sumPQ/n - mp * mq^T   (the identity noted above).
//   3. SVD H = U S V^T ;  R = V * diag(1,1,det(V*U^T)) * U^T ;  t = mq - R*mp.
// The det term flips the sign of the smallest singular direction if the naive
// solution came out as a reflection (det = -1) -- the standard Kabsch fix so R
// is a proper rotation. Runs on the host for BOTH CPU and GPU paths.
// ---------------------------------------------------------------------------
ICP_HD inline Rigid solve_rigid(const AccumFixed& a) {
    Rigid g = rigid_identity();
    if (a.count == 0) return g;                      // no pairs -> identity
    const double n = (double)a.count;

    double mp[3], mq[3];
    for (int i = 0; i < 3; ++i) {
        mp[i] = from_fixed(a.sumP[i], POS_SCALE) / n;   // centroid of moving set
        mq[i] = from_fixed(a.sumQ[i], POS_SCALE) / n;   // centroid of fixed set
    }

    // Centred cross-covariance H[r][c] = (1/n) sum p_r q_c  -  mp_r * mq_c.
    Mat3 H;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            H.m[r][c] = from_fixed(a.sumPQ[r * 3 + c], COV_SCALE) / n - mp[r] * mq[c];

    Mat3 U, V; double S[3];
    svd3x3(H, U, S, V);

    // R = V * U^T, then correct a possible reflection so det(R) = +1.
    Mat3 Ut = mat3_transpose(U);
    Mat3 R  = mat3_mul(V, Ut);
    if (mat3_det(R) < 0.0) {
        // Flip the sign of V's last column (the one tied to the smallest
        // singular value) and recompute -- the canonical Kabsch reflection fix.
        for (int r = 0; r < 3; ++r) V.m[r][2] = -V.m[r][2];
        R = mat3_mul(V, Ut);
    }

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) g.R[r][c] = R.m[r][c];
        // t = mq - R * mp.
        g.t[r] = mq[r] - (R.m[r][0]*mp[0] + R.m[r][1]*mp[1] + R.m[r][2]*mp[2]);
    }
    return g;
}

// ---------------------------------------------------------------------------
// centroid_prealign: the standard ICP INITIALIZATION. Return the rigid
// transform (identity rotation + a translation) that moves the moving cloud's
// centroid onto the fixed cloud's centroid:  t = mean(Q) - mean(P).
//
//   WHY THIS MATTERS (a real, load-bearing ICP lesson -- see THEORY "numerics"):
//   ICP is a LOCAL optimizer. Started from identity on a cloud that is offset
//   AND rotated, the very first nearest-neighbour correspondences are badly
//   wrong (each moving point latches onto the fixed point nearest its displaced
//   position), and ICP then converges to a poor local minimum (in our sample,
//   RMS stalls at ~3.6 mm instead of the ~0.2 mm noise floor). Cancelling the
//   gross translation first puts the two clouds on top of each other, so the
//   remaining problem is a modest rotation that ICP solves cleanly. This is why
//   every practical ICP begins with a coarse alignment (centroid match, or a
//   feature/landmark pre-registration). We use it as the starting guess g0.
//
//   Runs on the host (called once). Deterministic: a plain mean over the points.
// ---------------------------------------------------------------------------
ICP_HD inline Rigid centroid_prealign(const Vec3* P, int np, const Vec3* Q, int nq) {
    Rigid g = rigid_identity();
    if (np == 0 || nq == 0) return g;
    double mp[3] = {0,0,0}, mq[3] = {0,0,0};
    for (int i = 0; i < np; ++i) { mp[0]+=P[i].x; mp[1]+=P[i].y; mp[2]+=P[i].z; }
    for (int j = 0; j < nq; ++j) { mq[0]+=Q[j].x; mq[1]+=Q[j].y; mq[2]+=Q[j].z; }
    for (int k = 0; k < 3; ++k) {
        mp[k] /= (double)np;
        mq[k] /= (double)nq;
        g.t[k] = mq[k] - mp[k];      // translate P's centroid onto Q's
    }
    return g;
}

// Compose two rigid transforms: return  g2 ∘ g1  (apply g1 first, then g2).
// ICP updates the running transform each iteration by composing the newly
// solved incremental transform onto the accumulated one.
ICP_HD inline Rigid rigid_compose(const Rigid& g2, const Rigid& g1) {
    Rigid out;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) s += g2.R[r][k] * g1.R[k][c];
            out.R[r][c] = s;
        }
        out.t[r] = g2.R[r][0]*g1.t[0] + g2.R[r][1]*g1.t[1] + g2.R[r][2]*g1.t[2] + g2.t[r];
    }
    return out;
}
