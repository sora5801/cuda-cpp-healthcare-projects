// ===========================================================================
// src/tg43_physics.h  --  The ONE TRUE TG-43 dose formula (host + device)
// ---------------------------------------------------------------------------
// Project 5.7 : Brachytherapy Dose & Source Modeling
//
// THE SHARED __host__ __device__ CORE  (PATTERNS.md section 2)
//   This header holds the per-(voxel, dwell) dose-rate math -- the AAPM TG-43U1
//   formalism -- as plain inline functions decorated so that:
//     * the CPU reference (reference_cpu.cpp, compiled by cl.exe / g++) and
//     * the GPU kernel (kernels.cu, compiled by nvcc)
//   evaluate the *identical* arithmetic. That is what makes GPU-vs-CPU
//   verification meaningful: any disagreement is pure floating-point rounding
//   (FMA contraction), not a different algorithm.
//
//   RULE: keep this header free of CUDA-only constructs (no __global__, no
//   <<<>>>, no cudaXxx). Only the TG43_HD decorator, POD structs, and math.
//   That lets the plain host compiler include it too.
//
// WHAT TG-43 COMPUTES (the science lives in ../THEORY.md)
//   A sealed radioactive source (e.g. Ir-192 HDR) emits photons. Instead of
//   tracking those photons (that is project 5.1 / 5.10, Monte Carlo), the TG-43
//   protocol gives the *dose rate* at a point P(r, theta) around the source as a
//   product of pre-measured factors:
//
//     D_dot(r,theta) = S_K * Lambda
//                      * [ G_L(r,theta) / G_L(r0,theta0) ]   (geometry)
//                      * g_L(r)                               (radial, tabulated)
//                      * F(r,theta)                           (anisotropy, tabulated)
//
//   with the reference point r0 = 1 cm, theta0 = 90 deg (transverse axis).
//   r is the distance from the source center to P; theta is the polar angle
//   measured from the source long axis. G_L is the LINE-source geometry
//   function (the source has physical length L, not a mathematical point), and
//   g_L, F are interpolated from small measured tables kept in constant memory.
//
//   This header is included by reference_cpu.h (host) AND kernels.cu (device).
//   READ THIS AFTER: ../THEORY.md "The math"; BEFORE: reference_cpu.cpp, kernels.cu.
// ===========================================================================
#pragma once

#include <cmath>   // std::sqrt, std::atan2, std::fabs, M_PI-free constants

// ---------------------------------------------------------------------------
// TG43_HD: the "host+device" decorator idiom.
//   * Under nvcc (__CUDACC__ defined) every function is compiled for BOTH the
//     CPU and the GPU, so the kernel can call it on-device.
//   * Under a plain C++ compiler the decorator expands to nothing (those
//     keywords do not exist off-GPU), so reference_cpu.cpp still compiles.
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#define TG43_HD __host__ __device__
#else
#define TG43_HD
#endif

// pi as a compile-time constant. We avoid M_PI because it is not guaranteed by
// the C++ standard headers on MSVC without _USE_MATH_DEFINES; a literal is
// portable and identical on host and device.
#ifndef TG43_PI
#define TG43_PI 3.14159265358979323846
#endif

// Hard caps on table sizes so every struct below is Plain-Old-Data with fixed
// storage -- POD can be passed BY VALUE into a kernel (copied into the launch's
// parameter space) with no pointers to chase. Bump these if a source needs
// finer tables; they are deliberately generous for the teaching sample.
#define TG43_MAX_RADII   32   // rows in g_L(r) and in the F(r,theta) grid
#define TG43_MAX_ANGLES  24   // columns (theta samples) in the F(r,theta) grid
#define TG43_MAX_DWELLS  64   // dwell positions in one plan (HDR afterloader)

// ---------------------------------------------------------------------------
// SourceModel: everything that describes ONE brachytherapy source TYPE and its
// TG-43 consensus dataset. All arrays are fixed-size (POD) so the whole struct
// can be memcpy'd to the device or passed by value.
//
//   Units follow the TG-43 convention:
//     * lengths in centimeters (cm)
//     * S_K (air-kerma strength) in U = 1 uGy*m^2/h  (we carry it dimensionless
//       here and report dose rate in the same "cGy/h per U" scale as Lambda)
//     * Lambda (dose-rate constant) in cGy*h^-1*U^-1
// ---------------------------------------------------------------------------
struct SourceModel {
    double L;          // active source LENGTH in cm (line-source model). Ir-192
                       // HDR pellets are ~0.35 cm; a point source uses L->0.
    double Lambda;     // dose-rate constant [cGy/(h*U)]: dose rate at (r0,theta0)
                       // per unit air-kerma strength. Source-specific constant.

    // --- radial dose function g_L(r): accounts for absorption + scatter in
    //     water along the transverse axis, normalized so g_L(1 cm) = 1. ---
    int    n_g;                       // number of tabulated radii
    double g_r[TG43_MAX_RADII];       // radii [cm], strictly increasing
    double g_val[TG43_MAX_RADII];     // g_L at each radius (dimensionless, ~1 near 1cm)

    // --- 2D anisotropy function F(r,theta): how dose falls off away from the
    //     transverse plane (self-absorption in the source + capsule). Stored as
    //     a grid over (radius, angle); F(r,90deg) == 1 by definition. ---
    int    n_Fr;                      // rows: radii where F is tabulated
    int    n_Ft;                      // cols: polar angles where F is tabulated
    double F_r[TG43_MAX_RADII];       // the radii   [cm], increasing
    double F_t[TG43_MAX_ANGLES];      // the angles  [degrees], increasing (0..180)
    double F_val[TG43_MAX_RADII * TG43_MAX_ANGLES];  // row-major F[ir*n_Ft + it]
};

// ---------------------------------------------------------------------------
// Dwell: one source position in the plan. An HDR afterloader steps the single
// source along a catheter, pausing ("dwelling") at each position for a weighted
// time; the dose is the time-weighted superposition over all dwells.
//   (x,y,z) in cm; the source long axis is taken parallel to +z (a common
//   teaching simplification -- real plans orient each dwell along its catheter
//   tangent; see THEORY "Where this sits in the real world").
// ---------------------------------------------------------------------------
struct Dwell {
    double x, y, z;    // dwell center position [cm] in the dose grid frame
    double weight;     // relative dwell time * S_K  [U*h]; scales this source's
                       // contribution. Optimizing these weights is inverse
                       // planning (project 5.2); here they are given.
};

// ---------------------------------------------------------------------------
// clampd / lerp: tiny numeric helpers, defined once so host and device share
// them bit-for-bit. `lerp` is the standard linear interpolation a + t*(b-a).
// ---------------------------------------------------------------------------
TG43_HD inline double clampd(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
TG43_HD inline double lerp(double a, double b, double t) {
    return a + t * (b - a);
}

// ---------------------------------------------------------------------------
// interp1: linear interpolation of a 1-D table y(x) at query xq.
//   * xs must be strictly increasing (we do a simple linear scan -- n is tiny,
//     so a binary search would not pay for its branch complexity here).
//   * Outside the table we CLAMP to the endpoint value (flat extrapolation).
//     TG-43 tables are measured over the clinically relevant range; clamping is
//     the conventional, safe choice and keeps the function total.
//   Used for the radial dose function g_L(r).
// ---------------------------------------------------------------------------
TG43_HD inline double interp1(const double* xs, const double* ys, int n, double xq) {
    if (n <= 0) return 0.0;
    if (xq <= xs[0])      return ys[0];        // below table -> first value
    if (xq >= xs[n - 1])  return ys[n - 1];    // above table -> last value
    // Find the bracketing interval [xs[i], xs[i+1]] that contains xq.
    int i = 0;
    while (i < n - 1 && xs[i + 1] < xq) ++i;
    const double t = (xq - xs[i]) / (xs[i + 1] - xs[i]);  // fractional position
    return lerp(ys[i], ys[i + 1], t);
}

// ---------------------------------------------------------------------------
// find_bracket: return the lower index i such that grid[i] <= q < grid[i+1],
// plus the fractional position `frac` in [0,1] within that cell. Clamps at the
// ends (frac=0 below the grid, frac=1 above). Shared by the 2-D F(r,theta)
// bilinear interpolation for BOTH the radius axis and the angle axis.
// ---------------------------------------------------------------------------
TG43_HD inline int find_bracket(const double* grid, int n, double q, double* frac) {
    if (q <= grid[0])     { *frac = 0.0; return 0; }
    if (q >= grid[n - 1]) { *frac = 0.0; return n - 1; }   // frac 0 on last node
    int i = 0;
    while (i < n - 1 && grid[i + 1] < q) ++i;
    *frac = (q - grid[i]) / (grid[i + 1] - grid[i]);
    return i;
}

// ---------------------------------------------------------------------------
// anisotropy_F: bilinearly interpolate the 2-D anisotropy table F(r,theta).
//   F is stored row-major: F_val[ir * n_Ft + it]. We interpolate first along
//   theta within each of the two bracketing radius rows, then along r between
//   those two results -- the standard separable bilinear interpolation. On the
//   transverse plane (theta = 90 deg) a well-formed table returns ~1.
// ---------------------------------------------------------------------------
TG43_HD inline double anisotropy_F(const SourceModel& s, double r, double theta_deg) {
    double fr = 0.0, ft = 0.0;
    const int ir = find_bracket(s.F_r, s.n_Fr, r, &fr);          // radius cell
    const int it = find_bracket(s.F_t, s.n_Ft, theta_deg, &ft);  // angle cell
    const int ir1 = (ir + 1 < s.n_Fr) ? ir + 1 : ir;            // next radius row
    const int it1 = (it + 1 < s.n_Ft) ? it + 1 : it;            // next angle col
    // Four corners of the (r,theta) cell.
    const double f00 = s.F_val[ir  * s.n_Ft + it ];
    const double f01 = s.F_val[ir  * s.n_Ft + it1];
    const double f10 = s.F_val[ir1 * s.n_Ft + it ];
    const double f11 = s.F_val[ir1 * s.n_Ft + it1];
    const double top = lerp(f00, f01, ft);   // interpolate along theta at row ir
    const double bot = lerp(f10, f11, ft);   // interpolate along theta at row ir1
    return lerp(top, bot, fr);               // interpolate along r
}

// ---------------------------------------------------------------------------
// geometry_line: the TG-43 LINE-source geometry function G_L(r,theta).
//   Physical meaning: a source of length L is NOT a point; photons emanate from
//   along its length, so the pure inverse-square law is modified. For a line
//   source of length L:
//
//       G_L(r,theta) = beta / (L * r * sin(theta))        (theta != 0, 180)
//       G_L(r,theta) = 1 / (r^2 - L^2/4)                  (on the long axis)
//
//   where beta is the angle (in radians) subtended by the active length as seen
//   from P. We compute beta from the two triangle angles to the source ends.
//   As L -> 0 this collapses to the point-source 1/r^2, which is the sanity
//   check in THEORY "How we verify".
//
//   r      : distance source-center -> P [cm]  (must be > 0)
//   theta  : polar angle from the source long axis [degrees]
//   L      : active length [cm]
// ---------------------------------------------------------------------------
TG43_HD inline double geometry_line(double r, double theta_deg, double L) {
    // Point-source limit: no length -> exact inverse square. Also the safe path
    // when r is extremely small (avoid the line-source singularity at r->0).
    if (L <= 0.0) return 1.0 / (r * r);

    const double theta = theta_deg * (TG43_PI / 180.0);   // to radians
    const double sint  = std::sin(theta);
    const double half  = 0.5 * L;

    // On (or nearly on) the source long axis, sin(theta) ~ 0 and the general
    // formula divides by ~0. Use the closed-form long-axis expression, guarding
    // the case r <= L/2 (P inside the source extent) by flooring the denom.
    if (sint < 1.0e-6) {
        double denom = r * r - half * half;
        if (denom < 1.0e-9) denom = 1.0e-9;   // keep finite & positive
        return 1.0 / denom;
    }

    // General off-axis case. Place the source along z centered at the origin;
    // P is at (r sin(theta), 0, r cos(theta)). The angles from P to the two
    // source ends (z = +half, z = -half) give beta = theta2 - theta1.
    const double x  = r * sint;                 // perpendicular (transverse) offset
    const double z  = r * std::cos(theta);      // axial offset of P from center
    const double t1 = std::atan2(z + half, x);  // angle to the far end (+half)
    const double t2 = std::atan2(z - half, x);  // angle to the near end (-half)
    const double beta = t1 - t2;                // subtended angle [rad], > 0
    return beta / (L * x);                       // note L*x == L*r*sin(theta)
}

// ---------------------------------------------------------------------------
// dose_rate_one_dwell: the FULL TG-43 dose rate at world point P from ONE dwell.
//   This is the single function the CPU loops over and the GPU thread calls --
//   the heart of the whole project. Steps:
//     1. geometry: distance r and polar angle theta of P relative to the dwell
//        (dwell long axis assumed parallel to +z, per Dwell's note).
//     2. G_L ratio: G_L(r,theta) / G_L(r0, theta0) with r0=1cm, theta0=90deg.
//     3. g_L(r): radial dose function (1-D interpolation).
//     4. F(r,theta): anisotropy (2-D bilinear interpolation).
//     5. multiply by Lambda and the dwell weight (weight = S_K * dwell time).
//
//   Returns dose rate contribution in cGy/h (the plan sums these over dwells;
//   multiply by a treatment time upstream for absolute dose in cGy).
//
//   px,py,pz : voxel center position [cm]
// ---------------------------------------------------------------------------
TG43_HD inline double dose_rate_one_dwell(const SourceModel& s, const Dwell& d,
                                          double px, double py, double pz) {
    // --- 1. geometry of P relative to this dwell ---------------------------
    const double dx = px - d.x;
    const double dy = py - d.y;
    const double dz = pz - d.z;
    const double r  = std::sqrt(dx * dx + dy * dy + dz * dz);   // distance [cm]

    // Guard the source interior: dose rate is not defined AT the source. We
    // floor r to a tiny positive value so the function stays finite; voxels this
    // close are non-physical for TG-43 anyway (MBDCA/MC handle them -- THEORY).
    const double r_eff = (r < 1.0e-4) ? 1.0e-4 : r;

    // Polar angle theta from the source long axis (+z): cos(theta) = dz / r.
    // acos gives [0,180] degrees, exactly the F-table domain.
    const double cos_t   = clampd(dz / r_eff, -1.0, 1.0);
    const double theta_d = std::acos(cos_t) * (180.0 / TG43_PI);

    // --- 2. geometry-function RATIO (dimensionless) ------------------------
    // Reference geometry at r0 = 1 cm on the transverse axis (theta0 = 90 deg).
    const double G     = geometry_line(r_eff, theta_d, s.L);
    const double G_ref = geometry_line(1.0,   90.0,    s.L);
    const double G_ratio = G / G_ref;

    // --- 3. radial dose function g_L(r) ------------------------------------
    const double g = interp1(s.g_r, s.g_val, s.n_g, r_eff);

    // --- 4. anisotropy F(r,theta) ------------------------------------------
    const double F = anisotropy_F(s, r_eff, theta_d);

    // --- 5. assemble the TG-43 product -------------------------------------
    // D_dot = weight * Lambda * G_ratio * g_L * F.
    return d.weight * s.Lambda * G_ratio * g * F;
}
