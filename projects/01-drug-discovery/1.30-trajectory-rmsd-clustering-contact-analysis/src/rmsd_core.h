// ===========================================================================
// src/rmsd_core.h  --  The ONE TRUE per-frame physics, shared by CPU and GPU
// ---------------------------------------------------------------------------
// Project 1.30 : Trajectory RMSD, Clustering & Contact Analysis
//
// WHY THIS HEADER EXISTS  (PATTERNS.md sec 2 -- the HD-core idiom)
//   The single most useful trick in this repo: put the *per-element math* in ONE
//   header as `__host__ __device__` inline functions, so the CPU reference
//   (reference_cpu.cpp, compiled by cl.exe) and the GPU kernel (kernels.cu,
//   compiled by nvcc) run BYTE-FOR-BYTE IDENTICAL arithmetic. That turns
//   verification from "approximately agrees" into "agrees to ~machine epsilon"
//   and makes the CPU<->GPU diff a real correctness proof rather than a fuzzy
//   sanity check.
//
//   Everything here is plain C++/CUDA math on `double`s -- NO `__global__`, NO
//   CUDA-only types, NO std::vector -- so the host compiler can include it too.
//   The HD macro below expands to `__host__ __device__` under nvcc and to
//   nothing under the host compiler (which has never heard of those keywords).
//
// WHAT THE MATH IS  (see ../THEORY.md for the full derivation)
//   We solve two independent per-frame problems:
//
//   (1) KABSCH / QCP RMSD.  Given a moving frame X (N atoms x 3) and a fixed
//       reference Y, find the rotation R that minimizes the root-mean-square
//       deviation after optimal superposition:
//           RMSD = sqrt( min_R (1/N) * sum_i | R*(x_i - x_bar) - (y_i - y_bar) |^2 ).
//       The classic route is Kabsch's: build the 3x3 cross-covariance, take its
//       SVD, read off R. We instead use Theobald's QCP (Quaternion
//       Characteristic Polynomial, 2005) closed form, because it needs NO SVD --
//       just the largest root of a quartic -- which is trivial to make
//       deterministic and identical on CPU and GPU. The minimized RMSD is
//           RMSD = sqrt( max(0, (G_x + G_y - 2*lambda_max) / N) )
//       where G_x, G_y are the structures' inner products (Frobenius norms^2)
//       and lambda_max is the largest eigenvalue of a 4x4 symmetric "key" matrix
//       built from the 3x3 covariance M. This is exactly what MDTraj / the
//       `rmsd` library compute.
//
//   (2) FRACTION OF NATIVE CONTACTS  Q(frame).  Two atoms are "in contact" if
//       their distance is below a cutoff. The *native* contacts are the contacts
//       present in the reference frame. Q is the fraction of those native
//       contacts that survive in a given frame -- a standard 0..1 folding /
//       conformational-similarity coordinate. Per frame this is an O(N^2) sweep
//       over atom pairs, fully independent across frames.
//
// READ THIS BEFORE: reference_cpu.cpp and kernels.cu (both call these funcs).
// ===========================================================================
#pragma once

#include <math.h>     // sqrt, fabs, fmax, fmin, cos, acos  (the C math, HD-safe)

// ---------------------------------------------------------------------------
// HD: the host/device decorator. Under nvcc (__CUDACC__ defined) every function
// below is compiled for BOTH the CPU and the GPU. Under the plain host compiler
// the macro vanishes, so reference_cpu.cpp sees ordinary inline C++ functions.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define HD __host__ __device__
// UNROLL: emit `#pragma unroll` ONLY under nvcc. The host compiler (MSVC) does
// not know that pragma and would warn (C4068) when reference_cpu.cpp includes
// this header, so we route it through _Pragma and make it vanish on the host.
#define UNROLL _Pragma("unroll")
#else
#define HD
#define UNROLL
#endif

// Geometry is fixed at compile time so the math is simple and unrollable:
//   N_ATOMS atoms per frame, each with x,y,z. A frame is therefore a flat block
//   of N_ATOMS*3 doubles, laid out atom-major: [x0,y0,z0, x1,y1,z1, ...].
// 16 atoms keeps the committed sample tiny while still exercising a non-trivial
// 3x3 covariance and an O(N^2) contact sweep (16*16 = 256 pairs).
#ifndef N_ATOMS
#define N_ATOMS 16
#endif

// The contact cutoff (in the same length unit as the coordinates -- we use
// Angstrom-like arbitrary units for the synthetic data). A pair (i,j) with
// j > i + CONTACT_SEP is a contact iff |r_i - r_j| < CONTACT_CUTOFF.
#ifndef CONTACT_CUTOFF
#define CONTACT_CUTOFF 8.0
#endif
// Skip near-neighbours along the chain: trivially-close sequential atoms are not
// informative contacts (the standard "i, i+1, i+2 don't count" convention).
#ifndef CONTACT_SEP
#define CONTACT_SEP 2
#endif

// ---------------------------------------------------------------------------
// frame_ptr: address of frame f inside a flat trajectory buffer.
//   traj is [n_frames * N_ATOMS * 3] doubles, frame-major then atom-major.
//   Returned pointer points at the first coordinate (x of atom 0) of frame f.
// ---------------------------------------------------------------------------
HD inline const double* frame_ptr(const double* traj, int f) {
    return traj + (long long)f * (N_ATOMS * 3);
}

// ---------------------------------------------------------------------------
// A 3-vector centroid of one frame's N_ATOMS atoms (the geometric center we
// subtract before superposition -- translation is removed first so only the
// rotation remains to optimize). cx,cy,cz are out-params.
// ---------------------------------------------------------------------------
HD inline void frame_centroid(const double* fr, double* cx, double* cy, double* cz) {
    double sx = 0.0, sy = 0.0, sz = 0.0;
    for (int i = 0; i < N_ATOMS; ++i) {
        sx += fr[3 * i + 0];
        sy += fr[3 * i + 1];
        sz += fr[3 * i + 2];
    }
    *cx = sx / N_ATOMS;
    *cy = sy / N_ATOMS;
    *cz = sz / N_ATOMS;
}

// ---------------------------------------------------------------------------
// kabsch_rmsd: optimal-superposition RMSD between a moving frame and a fixed
// reference, via Theobald's QCP closed form. NO SVD, NO matrix factorization --
// just a 4x4 eigenvalue found by Newton's method on the characteristic
// polynomial. Deterministic and identical on CPU and GPU.
//
//   fr   : pointer to the moving frame   (N_ATOMS*3 doubles)
//   ref  : pointer to the reference frame (N_ATOMS*3 doubles)
//   returns: the minimized RMSD (>= 0).
//
//   STEPS (each labeled below):
//     A. Center both structures on their centroids (remove translation).
//     B. Accumulate G = G_x + G_y (sum of squared coordinates, both centered)
//        and the 3x3 cross-covariance M = sum_i x_i (y_i)^T.
//     C. Build the 4x4 symmetric "key" matrix K(M) and its characteristic
//        quartic; QCP gives the coefficients in closed form from M's entries.
//     D. Find lambda_max (the largest root) by Newton iteration started at G/2,
//        which is a rigorous upper bound for lambda_max (so Newton descends to
//        the largest root monotonically -- a known QCP property).
//     E. RMSD = sqrt( max(0, (G - 2*lambda_max)/N) ).
//
//   We keep a FIXED iteration count (no data-dependent break) so the CPU and GPU
//   execute the identical sequence of floating-point ops -> bit-identical result.
// ---------------------------------------------------------------------------
HD inline double kabsch_rmsd(const double* fr, const double* ref) {
    // --- A. centroids ------------------------------------------------------
    double cx, cy, cz, rx, ry, rz;
    frame_centroid(fr,  &cx, &cy, &cz);
    frame_centroid(ref, &rx, &ry, &rz);

    // --- B. inner products G and 3x3 covariance M --------------------------
    // M is the cross-covariance sum_i (moving_i) outer (reference_i).
    double Sxx = 0, Sxy = 0, Sxz = 0;
    double Syx = 0, Syy = 0, Syz = 0;
    double Szx = 0, Szy = 0, Szz = 0;
    double G = 0.0;   // G_x + G_y : Frobenius norms^2 of both centered structures
    for (int i = 0; i < N_ATOMS; ++i) {
        const double ax = fr[3 * i + 0] - cx;   // centered moving atom i
        const double ay = fr[3 * i + 1] - cy;
        const double az = fr[3 * i + 2] - cz;
        const double bx = ref[3 * i + 0] - rx;  // centered reference atom i
        const double by = ref[3 * i + 1] - ry;
        const double bz = ref[3 * i + 2] - rz;
        G += ax * ax + ay * ay + az * az + bx * bx + by * by + bz * bz;
        Sxx += ax * bx; Sxy += ax * by; Sxz += ax * bz;
        Syx += ay * bx; Syy += ay * by; Syz += ay * bz;
        Szx += az * bx; Szy += az * by; Szz += az * bz;
    }

    // --- C. Build the 4x4 symmetric "key" matrix K(M) ----------------------
    // The QCP method maps the optimal-rotation problem to: find the LARGEST
    // eigenvalue of a 4x4 symmetric matrix K built from the 3x3 covariance M.
    // (Its eigenvector is the optimal-rotation quaternion -- which we don't need
    // for RMSD, only the eigenvalue.) The standard K (Theobald 2005, Eq. for the
    // key matrix; same matrix used by Horn's quaternion method) is:
    //
    //   K = [ Sxx+Syy+Szz   Syz-Szy        Szx-Sxz        Sxy-Syx      ]
    //       [ Syz-Szy       Sxx-Syy-Szz    Sxy+Syx        Szx+Sxz      ]
    //       [ Szx-Sxz       Sxy+Syx       -Sxx+Syy-Szz    Syz+Szy      ]
    //       [ Sxy-Syx       Szx+Sxz        Syz+Szy       -Sxx-Syy+Szz  ]
    //
    // K is symmetric and TRACELESS (the diagonal sums to 0), so its
    // characteristic polynomial has no cubic term: p(l) = l^4 + c2 l^2 + c1 l + c0.
    // We store K as a flat 4x4 (row-major) and derive c2,c1,c0 exactly below.
    double K[16];
    K[0]  = Sxx + Syy + Szz; K[1]  = Syz - Szy;        K[2]  = Szx - Sxz;        K[3]  = Sxy - Syx;
    K[4]  = Syz - Szy;       K[5]  = Sxx - Syy - Szz;  K[6]  = Sxy + Syx;        K[7]  = Szx + Sxz;
    K[8]  = Szx - Sxz;       K[9]  = Sxy + Syx;        K[10] = -Sxx + Syy - Szz; K[11] = Syz + Szy;
    K[12] = Sxy - Syx;       K[13] = Szx + Sxz;        K[14] = Syz + Szy;        K[15] = -Sxx - Syy + Szz;

    // Faddeev-LeVerrier: derive the characteristic-polynomial coefficients of a
    // 4x4 matrix EXACTLY from traces of its powers -- only matrix multiplies and
    // traces, no factorization, so it is auditable and bit-identical CPU vs GPU.
    // For a traceless 4x4 the polynomial is  l^4 + c2 l^2 + c1 l + c0  with
    //   c2 = -1/2 * tr(K^2)
    //   c1 = -1/3 * tr(K^3)                    (since tr(K)=0)
    //   c0 =  det(K) = (1/8) tr(K^2)^2 ... -> we get it cleanly from L-V below.
    // We run the L-V recurrence p1..p4 with c_k accumulated; it is short enough
    // to read in full. (Indices: a[r*4+c].)
    double K2[16];   // K*K
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) {
            double s = 0.0;
            for (int k = 0; k < 4; ++k) s += K[r * 4 + k] * K[k * 4 + c];
            K2[r * 4 + c] = s;
        }
    double K3[16];   // K*K2
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) {
            double s = 0.0;
            for (int k = 0; k < 4; ++k) s += K[r * 4 + k] * K2[k * 4 + c];
            K3[r * 4 + c] = s;
        }
    double K4[16];   // K*K3
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) {
            double s = 0.0;
            for (int k = 0; k < 4; ++k) s += K[r * 4 + k] * K3[k * 4 + c];
            K4[r * 4 + c] = s;
        }
    const double t1 = K[0]  + K[5]  + K[10] + K[15];   // tr(K)   (= 0, kept for clarity)
    const double t2 = K2[0] + K2[5] + K2[10] + K2[15]; // tr(K^2)
    const double t3 = K3[0] + K3[5] + K3[10] + K3[15]; // tr(K^3)
    const double t4 = K4[0] + K4[5] + K4[10] + K4[15]; // tr(K^4)
    // Newton's identities give the elementary symmetric polynomials e_k of the
    // eigenvalues; the monic char-poly is l^4 - e1 l^3 + e2 l^2 - e3 l + e4.
    const double e1 = t1;                                  // = 0
    const double e2 = 0.5 * (e1 * t1 - t2);
    const double e3 = (1.0 / 3.0) * (e2 * t1 - e1 * t2 + t3);
    const double e4 = 0.25 * (e3 * t1 - e2 * t2 + e1 * t3 - t4);
    // So p(l) = l^4 + C3 l^3 + C2 l^2 + C1 l + C0 with the signs below.
    const double C3 = -e1;     // 0
    const double C2 =  e2;
    const double C1 = -e3;
    const double C0 =  e4;

    // --- D. lambda_max by Newton on p(l) -----------------------------------
    // G/2 is a rigorous upper bound for lambda_max (the largest eigenvalue of K
    // is <= Gx/... ; G/2 is safely above it), and p is monotone increasing beyond
    // the largest root, so Newton from G/2 converges DOWN to lambda_max. We run a
    // FIXED 50 iterations (ample for double precision) with no data-dependent
    // break, so CPU and GPU trace identical arithmetic -> identical result.
    double lambda = 0.5 * G;
    UNROLL
    for (int it = 0; it < 50; ++it) {
        const double l2 = lambda * lambda;
        const double p  = (l2 * l2) + C3 * l2 * lambda + C2 * l2 + C1 * lambda + C0;
        const double dp = 4.0 * l2 * lambda + 3.0 * C3 * l2 + 2.0 * C2 * lambda + C1;
        // Guard a vanishing derivative (would divide by ~0); if so, stop moving.
        const double step = (fabs(dp) > 1e-300) ? (p / dp) : 0.0;
        lambda -= step;
    }

    // --- E. RMSD from lambda_max -------------------------------------------
    double msd = (G - 2.0 * lambda) / (double)N_ATOMS;   // mean-square deviation
    if (msd < 0.0) msd = 0.0;                            // clamp tiny negatives
    return sqrt(msd);
}

// ---------------------------------------------------------------------------
// count_native_contacts: number of contacts in a single (reference) frame.
//   A contact is a pair (i, j), j > i + CONTACT_SEP, with distance < cutoff.
//   This O(N^2) sweep defines the *native* contact set used by frac_native_*.
// ---------------------------------------------------------------------------
HD inline int count_native_contacts(const double* ref) {
    const double cut2 = (double)CONTACT_CUTOFF * (double)CONTACT_CUTOFF;
    int total = 0;
    for (int i = 0; i < N_ATOMS; ++i) {
        for (int j = i + CONTACT_SEP + 1; j < N_ATOMS; ++j) {
            const double dx = ref[3 * i + 0] - ref[3 * j + 0];
            const double dy = ref[3 * i + 1] - ref[3 * j + 1];
            const double dz = ref[3 * i + 2] - ref[3 * j + 2];
            if (dx * dx + dy * dy + dz * dz < cut2) ++total;
        }
    }
    return total;
}

// ---------------------------------------------------------------------------
// frac_native_contacts: Q(frame) -- the fraction of the reference's native
// contacts that are also present in `fr`. Returns a value in [0, 1].
//   We recompute "is this native pair still in contact?" for `fr` using the
//   SAME cutoff, so Q=1 when fr==ref and decreases as the structure changes.
//   native_total is passed in (computed once from the reference) to avoid
//   recomputing it per frame.
// ---------------------------------------------------------------------------
HD inline double frac_native_contacts(const double* fr, const double* ref,
                                      int native_total) {
    if (native_total <= 0) return 0.0;   // no native contacts -> define Q = 0
    const double cut2 = (double)CONTACT_CUTOFF * (double)CONTACT_CUTOFF;
    int kept = 0;
    for (int i = 0; i < N_ATOMS; ++i) {
        for (int j = i + CONTACT_SEP + 1; j < N_ATOMS; ++j) {
            // Only native pairs (contact in the reference) can be "kept".
            const double rdx = ref[3 * i + 0] - ref[3 * j + 0];
            const double rdy = ref[3 * i + 1] - ref[3 * j + 1];
            const double rdz = ref[3 * i + 2] - ref[3 * j + 2];
            if (rdx * rdx + rdy * rdy + rdz * rdz >= cut2) continue;  // not native
            // Is this native pair still in contact in the moving frame?
            const double dx = fr[3 * i + 0] - fr[3 * j + 0];
            const double dy = fr[3 * i + 1] - fr[3 * j + 1];
            const double dz = fr[3 * i + 2] - fr[3 * j + 2];
            if (dx * dx + dy * dy + dz * dz < cut2) ++kept;
        }
    }
    return (double)kept / (double)native_total;
}
