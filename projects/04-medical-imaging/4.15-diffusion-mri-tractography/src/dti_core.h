// ===========================================================================
// src/dti_core.h  --  Shared (host + device) DTI per-voxel math core
// ---------------------------------------------------------------------------
// Project 4.15 : Diffusion MRI & Tractography
//
// WHAT THIS FILE IS
//   The single source of truth for the *per-voxel physics* of Diffusion Tensor
//   Imaging (DTI). Everything here is marked __host__ __device__ (via the DTI_HD
//   macro) so the CPU reference (reference_cpu.cpp) and the GPU kernel
//   (kernels.cu) run BYTE-FOR-BYTE IDENTICAL math. That is the PATTERNS.md §2
//   idiom: putting the element-wise computation in one header makes verification
//   *exact* instead of approximate, and removes any chance of the two paths
//   drifting apart. Keep CUDA-only constructs (no __global__, no <cuda_runtime>)
//   out of this header so the plain host compiler can include it too.
//
// THE SCIENCE (see ../THEORY.md for the full derivation)
//   In diffusion MRI we measure how freely water diffuses along many directions.
//   For each diffusion-weighted measurement k we apply a gradient direction
//   g_k = (gx,gy,gz) with "b-value" b_k (units s/mm^2, encodes gradient strength
//   and timing). The Stejskal-Tanner equation predicts the measured signal:
//
//       S_k = S0 * exp( -b_k * g_k^T D g_k )
//
//   where S0 is the non-diffusion-weighted (b=0) signal and D is the 3x3
//   symmetric positive-definite DIFFUSION TENSOR of that voxel (units mm^2/s). D
//   has 6 unique entries:  Dxx Dyy Dzz Dxy Dxz Dyz.
//
//   Taking logs LINEARISES the model:
//
//       ln(S_k) = ln(S0) - b_k ( gx^2 Dxx + gy^2 Dyy + gz^2 Dzz
//                                + 2 gx gy Dxy + 2 gx gz Dxz + 2 gy gz Dyz )
//
//   Stack all measurements: y_k = ln(S_k) is a linear function of the 7 unknowns
//   ( ln(S0), Dxx, Dyy, Dzz, Dxy, Dxz, Dyz ). With >= 7 measurements this is an
//   ordinary least-squares (OLS) fit  d = M y,  where M = (B^T B)^{-1} B^T is the
//   fixed pseudo-inverse of the design matrix B (SAME for every voxel because the
//   gradient scheme is shared). We compute M ONCE on the host and reuse it for
//   all N voxels -- so the per-voxel kernel is just a small fixed matrix-vector
//   product plus an eigen-decomposition. That is what makes DTI embarrassingly
//   parallel: one independent thread per voxel.
//
//   From D we derive the two headline scalar maps and the fiber direction:
//     * eigenvalues  l1 >= l2 >= l3 of D  (principal diffusivities),
//     * MD  = (l1+l2+l3)/3                 (Mean Diffusivity),
//     * FA  = sqrt(3/2) * ||L - MD*I|| / ||L||  in [0,1] (Fractional Anisotropy;
//            0 = isotropic diffusion, ->1 = strongly directional white matter),
//     * v1 = eigenvector of l1            (the local fiber orientation, used by
//            the deterministic tractography in kernels.cu).
//
// READ THIS BEFORE: reference_cpu.h, kernels.cuh. The math is in ../THEORY.md.
// ===========================================================================
#pragma once

#include <cmath>     // std::exp, std::log, std::sqrt, std::acos, std::cos, std::fabs

// DTI_HD expands to __host__ __device__ when compiled by nvcc, and to nothing
// under the host C++ compiler (which does not know those keywords). This is the
// HD-macro idiom from PATTERNS.md §2 -- the one trick that keeps CPU and GPU in
// lockstep.
#ifdef __CUDACC__
#define DTI_HD __host__ __device__
// DTI_UNROLL asks nvcc to fully unroll a fixed-count loop (the same "#pragma
// unroll" you would write directly). We route it through a macro so the HOST
// compiler -- which does not understand "#pragma unroll" and would emit warning
// C4068 -- sees nothing. _Pragma() is the C99 operator form of #pragma, usable
// inside a macro. Keeping these HD headers warning-clean matters because the
// host reference (reference_cpu.cpp) includes them (CLAUDE.md §9: warnings are
// defects).
#define DTI_UNROLL _Pragma("unroll")
#else
#define DTI_HD
#define DTI_UNROLL
#endif

// ---------------------------------------------------------------------------
// FIXED ACQUISITION GEOMETRY (compile-time constants)
//   NDIR   : number of DIFFUSION-WEIGHTED directions (b > 0). We use a 12-point
//            electrostatic-repulsion scheme (a small, well-conditioned set that
//            is enough to fit the 6-parameter tensor robustly; real scans use
//            30-64+). Directions are defined in reference_cpu.cpp.
//   NMEAS  : total measurements = 1 (the b=0 image) + NDIR. Row 0 is always b=0.
//   NPARAM : the 7 linear unknowns  [ln S0, Dxx, Dyy, Dzz, Dxy, Dxz, Dyz].
//   Making these compile-time constants lets the fixed-size design matrix M live
//   in GPU constant memory and lets the inner loops fully unroll.
// ---------------------------------------------------------------------------
static constexpr int NDIR   = 12;             // diffusion-weighted directions
static constexpr int NMEAS  = 1 + NDIR;       // = 13 measurements per voxel
static constexpr int NPARAM = 7;              // ln(S0) + 6 tensor components

// A compact result bundle for one voxel: the scalar maps + principal direction.
// Returned by fit_voxel() and compared (CPU vs GPU) in main.cu.
struct VoxelResult {
    double md;        // mean diffusivity  (mm^2/s)
    double fa;        // fractional anisotropy in [0,1] (dimensionless)
    double l1;        // largest eigenvalue  (principal diffusivity, mm^2/s)
    double l2;        // middle eigenvalue
    double l3;        // smallest eigenvalue
    double v1x, v1y, v1z;   // principal eigenvector (unit fiber direction)
};

// ---------------------------------------------------------------------------
// clamp_unit: keep a value inside [-1, +1].
//   std::acos is only defined on [-1,1]; tiny floating-point overshoot (e.g.
//   1.0000000002) would otherwise produce NaN. We clamp defensively. This is a
//   classic numerical-robustness guard (see THEORY "Numerical considerations").
// ---------------------------------------------------------------------------
DTI_HD inline double clamp_unit(double x) {
    if (x >  1.0) return  1.0;
    if (x < -1.0) return -1.0;
    return x;
}

// ---------------------------------------------------------------------------
// sym3_eigen_analytic: closed-form eigen-decomposition of a 3x3 SYMMETRIC matrix.
//
//   WHY ANALYTIC (not cuSOLVER)?  The tensor is only 3x3 and symmetric, so its
//   eigenvalues are the three real roots of the characteristic cubic, given by
//   Smith's trigonometric formula (1961). This is BRANCH-FREE, deterministic,
//   and needs no library -- ideal for one-thread-per-voxel work where launching
//   a batched dense solver would be overkill. THEORY "GPU mapping" explains when
//   you WOULD reach for cuSOLVER (larger/general matrices). The result is
//   identical on host and device because the arithmetic is identical.
//
//   Input: the 6 unique components of the symmetric matrix
//          [ a  d  e ]
//          [ d  b  f ]
//          [ e  f  c ]
//   Output: eigenvalues w[0] >= w[1] >= w[2] (sorted descending) and the unit
//           eigenvector for the LARGEST eigenvalue in (v1x,v1y,v1z).
//
//   Algorithm (Smith 1961):
//     q  = trace/3
//     p2 = sum of squares of (A - q I) ; p = sqrt(p2/6)
//     B  = (A - q I)/p  (has det in [-1,1]); phi = acos(det(B)/2)/3
//     eigenvalues = q + 2 p cos(phi + k*2pi/3), k = 0,1,2.
// ---------------------------------------------------------------------------
DTI_HD inline void sym3_eigen_analytic(double a, double b, double c,
                                       double d, double e, double f,
                                       double w[3],
                                       double& v1x, double& v1y, double& v1z) {
    // p1 measures how far the matrix is from diagonal (sum of squared off-diags).
    const double p1 = d*d + e*e + f*f;
    const double q  = (a + b + c) / 3.0;          // mean of the diagonal = trace/3

    if (p1 <= 0.0) {
        // Already diagonal: eigenvalues are the diagonal entries themselves.
        w[0] = a; w[1] = b; w[2] = c;
    } else {
        // p2 = ||A - qI||_F^2 ; the trigonometric root formula follows.
        const double da = a - q, db = b - q, dc = c - q;
        const double p2 = da*da + db*db + dc*dc + 2.0*p1;
        const double p  = std::sqrt(p2 / 6.0);
        // det( (A - qI)/p ) / 2, clamped so acos() is always valid.
        const double bxx = da / p, byy = db / p, bzz = dc / p;
        const double bxy = d / p,  bxz = e / p,  byz = f / p;
        const double detB = bxx*(byy*bzz - byz*byz)
                          - bxy*(bxy*bzz - byz*bxz)
                          + bxz*(bxy*byz - byy*bxz);
        const double phi = std::acos(clamp_unit(detB / 2.0)) / 3.0;
        // The three roots are equally spaced by 2pi/3 around the circle.
        const double PI = 3.14159265358979323846;
        w[0] = q + 2.0 * p * std::cos(phi);
        w[2] = q + 2.0 * p * std::cos(phi + (2.0/3.0) * PI);
        w[1] = 3.0 * q - w[0] - w[2];              // trace is invariant
    }

    // SORT the three eigenvalues DESCENDING with an explicit 3-element sorting
    // network (three compare-swaps). WHY THIS MATTERS (a real numerical lesson):
    // Smith's formula *usually* returns them ordered, but when two eigenvalues
    // are (nearly) DEGENERATE -- e.g. the two "across-fiber" diffusivities of an
    // axially-symmetric tensor -- a difference of ~1e-16 in acos()/cos() between
    // the host math library and the device can FLIP which root is labelled the
    // largest. That is a *labeling* difference, not a real error (FA and MD are
    // symmetric in the eigenvalues, so they stay identical), but it makes l1/l2/l3
    // disagree between CPU and GPU. Explicitly sorting removes the ambiguity so
    // l1 >= l2 >= l3 deterministically on both. (THEORY "Numerical considerations".)
    if (w[0] < w[1]) { double t = w[0]; w[0] = w[1]; w[1] = t; }
    if (w[1] < w[2]) { double t = w[1]; w[1] = w[2]; w[2] = t; }
    if (w[0] < w[1]) { double t = w[0]; w[0] = w[1]; w[1] = t; }

    // --- Eigenvector for the LARGEST eigenvalue w[0] --------------------------
    // For a symmetric matrix, (A - w0 I) is rank <= 2; its null space is the
    // eigenvector. We take the cross product of two rows of (A - w0 I): the
    // cross product of any two independent rows is orthogonal to both, hence in
    // the null space. We pick the largest of the three candidate cross products
    // for numerical stability (avoids the degenerate near-zero case).
    const double l = w[0];
    // Rows of (A - lI):
    const double r0x = a - l, r0y = d,     r0z = e;
    const double r1x = d,     r1y = b - l, r1z = f;
    const double r2x = e,     r2y = f,     r2z = c - l;

    // Three candidate eigenvectors = cross products of row pairs.
    double cx[3], cy[3], cz[3], mag[3];
    cx[0] = r0y*r1z - r0z*r1y; cy[0] = r0z*r1x - r0x*r1z; cz[0] = r0x*r1y - r0y*r1x;
    cx[1] = r0y*r2z - r0z*r2y; cy[1] = r0z*r2x - r0x*r2z; cz[1] = r0x*r2y - r0y*r2x;
    cx[2] = r1y*r2z - r1z*r2y; cy[2] = r1z*r2x - r1x*r2z; cz[2] = r1x*r2y - r1y*r2x;
    for (int i = 0; i < 3; ++i) mag[i] = cx[i]*cx[i] + cy[i]*cy[i] + cz[i]*cz[i];

    int best = 0;
    if (mag[1] > mag[best]) best = 1;
    if (mag[2] > mag[best]) best = 2;

    double vx = cx[best], vy = cy[best], vz = cz[best];
    double nrm = std::sqrt(mag[best]);
    if (nrm <= 0.0) {
        // Fully isotropic voxel: no preferred direction. Return +x by convention
        // so downstream code never divides by zero (FA will be ~0 anyway).
        vx = 1.0; vy = 0.0; vz = 0.0; nrm = 1.0;
    }
    // Sign convention: force a non-negative dominant component so the direction
    // is deterministic (an eigenvector and its negation are both valid; we must
    // pick ONE so CPU and GPU agree and tractography does not flip randomly).
    double sx = vx / nrm, sy = vy / nrm, sz = vz / nrm;
    // Find the dominant axis and make its sign positive.
    double ax = std::fabs(sx), ay = std::fabs(sy), az = std::fabs(sz);
    double dom = (ax >= ay && ax >= az) ? sx : (ay >= az ? sy : sz);
    if (dom < 0.0) { sx = -sx; sy = -sy; sz = -sz; }
    v1x = sx; v1y = sy; v1z = sz;
}

// ---------------------------------------------------------------------------
// tensor_scalars: turn eigenvalues into MD and FA (the two headline maps).
//   MD = mean of the eigenvalues.
//   FA = sqrt(1/2) * sqrt( (l1-l2)^2 + (l2-l3)^2 + (l3-l1)^2 ) / sqrt(l1^2+l2^2+l3^2)
//      which is the standard Basser-Pierpaoli formula, bounded in [0,1].
//   FA is the single most-used dMRI scalar: near 0 in gray matter / CSF (water
//   diffuses equally in all directions) and high along white-matter tracts.
// ---------------------------------------------------------------------------
DTI_HD inline void tensor_scalars(const double w[3], double& md, double& fa) {
    md = (w[0] + w[1] + w[2]) / 3.0;
    const double num = (w[0]-w[1])*(w[0]-w[1])
                     + (w[1]-w[2])*(w[1]-w[2])
                     + (w[2]-w[0])*(w[2]-w[0]);
    const double den = w[0]*w[0] + w[1]*w[1] + w[2]*w[2];
    // Guard the all-zero (empty/background) voxel so FA is 0 rather than NaN.
    fa = (den > 0.0) ? std::sqrt(0.5 * num / den) : 0.0;
}

// ---------------------------------------------------------------------------
// fit_voxel: the COMPLETE per-voxel pipeline, shared by CPU and GPU.
//
//   Inputs:
//     signal : NMEAS raw signal intensities for this voxel (signal[0] = b=0).
//     Minv   : the NPARAM x NMEAS OLS pseudo-inverse (row-major), precomputed on
//              the host from the fixed gradient scheme. d = Minv * y where
//              y_k = ln(signal_k). Minv is the SAME for every voxel.
//   Output:
//     the VoxelResult (MD, FA, eigenvalues, principal direction).
//
//   Steps: (1) log-transform the signals, (2) OLS solve for [lnS0, 6 tensor
//   comps] via the fixed matvec, (3) eigen-decompose the tensor, (4) derive
//   MD/FA. Every step is deterministic double-precision arithmetic, so the
//   host and device produce identical bits (THEORY "How we verify").
// ---------------------------------------------------------------------------
DTI_HD inline VoxelResult fit_voxel(const double* signal, const double* Minv) {
    // (1) Linearise: y_k = ln(S_k). We floor the signal at a tiny positive value
    //     so a zero/negative measurement (noise) does not produce -inf/NaN.
    double y[NMEAS];
    DTI_UNROLL
    for (int k = 0; k < NMEAS; ++k) {
        double s = signal[k];
        if (s < 1e-12) s = 1e-12;     // clamp: log needs a positive argument
        y[k] = std::log(s);
    }

    // (2) OLS fit: params = Minv (NPARAM x NMEAS) * y (NMEAS). params[0] = ln(S0)
    //     is not needed downstream; params[1..6] are the 6 tensor components.
    double params[NPARAM];
    DTI_UNROLL
    for (int p = 0; p < NPARAM; ++p) {
        double acc = 0.0;
        DTI_UNROLL
        for (int k = 0; k < NMEAS; ++k)
            acc += Minv[p * NMEAS + k] * y[k];
        params[p] = acc;
    }
    // Unpack the symmetric tensor components (order matches the design matrix
    // columns defined in reference_cpu.cpp::build_design_matrix).
    const double Dxx = params[1], Dyy = params[2], Dzz = params[3];
    const double Dxy = params[4], Dxz = params[5], Dyz = params[6];

    // (3) Eigen-decompose D = [[Dxx,Dxy,Dxz],[Dxy,Dyy,Dyz],[Dxz,Dyz,Dzz]].
    double w[3], v1x, v1y, v1z;
    sym3_eigen_analytic(Dxx, Dyy, Dzz, Dxy, Dxz, Dyz, w, v1x, v1y, v1z);

    // (4) Scalar maps.
    VoxelResult out;
    tensor_scalars(w, out.md, out.fa);
    out.l1 = w[0]; out.l2 = w[1]; out.l3 = w[2];
    out.v1x = v1x; out.v1y = v1y; out.v1z = v1z;
    return out;
}
