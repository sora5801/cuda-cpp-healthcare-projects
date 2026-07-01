// ===========================================================================
// src/ccc_physics.h  --  Shared (host + device) physics for collapsed-cone SC dose
// ---------------------------------------------------------------------------
// Project 5.4 : Collapsed-Cone / Superposition-Convolution Dose  (2-D teaching model)
//
// WHY THIS HEADER IS SHARED  (the HD-macro idiom, PATTERNS.md §2)
//   The whole point of the verification in this project is that the CPU
//   reference (reference_cpu.cpp) and the GPU kernels (kernels.cu) run the
//   *identical* arithmetic, so their dose grids match EXACTLY. That only works
//   if both sides call the SAME functions. So every per-voxel formula lives here,
//   in ONE header, included by reference_cpu.cpp (plain host compiler) AND by
//   kernels.cu / main.cu (nvcc). The CCC_HD macro expands to `__host__ __device__`
//   under nvcc and to nothing under the host compiler, so the same `inline`
//   functions compile in both worlds. Keep CUDA-only constructs (`__global__`,
//   `blockIdx`, ...) OUT of this header so the host compiler can include it.
//
// WHAT THIS FILE MODELS  (a deliberately reduced 2-D teaching version; the full
// 3-D / ~400-cone production picture is in ../THEORY.md "Where this sits in the
// real world")
//
//   Superposition-convolution (SC) dose has two stages, and this header holds the
//   physics for both:
//
//   STAGE 1 -- TERMA (Total Energy Released per unit MAss).
//     A photon beam enters the TOP of a 2-D density grid (rho[y][x], units of
//     g/cm^3 relative to water). Primary photon fluence Psi attenuates as it
//     travels DOWN each column according to the *radiological* path length
//     (the density-weighted distance), i.e. Beer-Lambert with heterogeneity:
//         Psi(depth) = Psi0 * exp( -(mu/rho) * integral( rho dl ) ).
//     TERMA in a voxel is the energy that photon interactions release there:
//         T = (mu/rho) * Psi.
//     This is the Siddon-style ray-trace: one ray per beam column, marching voxel
//     by voxel accumulating the radiological depth. In this 2-D model the beam is
//     axis-aligned (straight down), so "Siddon" reduces to a clean vertical march
//     -- THEORY.md explains the general oblique-ray Siddon algorithm.
//
//   STAGE 2 -- Collapsed-cone convolution (CCC).
//     The energy released as TERMA does NOT deposit locally: secondary electrons
//     and scattered photons carry it away, spreading dose around each interaction
//     site per a Monte-Carlo-derived "dose-spread kernel". CCC approximates that
//     kernel by collapsing it onto a small set of discrete CONE directions. Along
//     each cone direction, dose transport becomes a 1-D exponential recursion:
//         D[k] = D[k-1] * exp(-a * d_rad) + T[k] * (1 - exp(-a * d_rad)) / a-ish,
//     where d_rad is the density-scaled step length (so the kernel reaches FARTHER
//     in low-density lung and LESS in dense bone -- the heterogeneity correction).
//     We use the standard analytic "transport + build-up" collapsed-cone update
//     below. Summing the contributions of all cone directions reconstructs the
//     full scatter kernel: that sum-over-cones IS the "superposition".
//
// DETERMINISM  (PATTERNS.md §3)
//   Multiple cones deposit into the SAME dose voxel, so on the GPU we scatter with
//   atomicAdd. Floating-point atomicAdd is NOT associative -> nondeterministic. So
//   we quantize each voxel contribution to an INTEGER number of "dose units"
//   (dose_to_units) and atomicAdd those; integer adds commute, so the GPU grid is
//   bit-identical to the CPU grid AND reproducible run to run. dose_units are
//   converted back to a physical-ish "dose" only for display.
// ===========================================================================
#pragma once

#include <cstdint>
#include <cmath>

// CCC_HD: __host__ __device__ under nvcc, nothing under the plain host compiler.
#ifdef __CUDACC__
#define CCC_HD __host__ __device__
#else
#define CCC_HD
#endif

// ---------------------------------------------------------------------------
// The problem geometry + physics parameters (read from the sample file).
//   All lengths are in centimetres; density rho is relative to water (water=1).
//   The grid is row-major: voxel (x=col, y=row) lives at index  y*nx + x, with
//   y=0 the TOP row where the beam enters and y increasing DOWNWARD (into depth).
// ---------------------------------------------------------------------------
struct CccParams {
    int    nx;            // grid width  (voxels across the beam)
    int    ny;            // grid height (voxels along the beam depth)
    double voxel_cm;      // physical edge length of one square voxel (cm)
    double mu_over_rho;   // mass attenuation coefficient mu/rho (cm^2/g) -- how
                          // strongly the medium interacts per unit areal density
    double psi0;          // incident primary fluence at the top surface (arb. units)
    int    n_cones;       // number of collapsed-cone directions (see cone tables)
    double kernel_a;      // cone kernel decay constant a (1/cm of *radiological*
                          // path) -- larger a => dose deposits closer to TERMA
    double dose_scale;    // fixed-point quantization: dose_units = round(D * dose_scale)
    int    beam_x0;       // first column irradiated by the beam (inclusive)
    int    beam_x1;       // last  column irradiated by the beam (inclusive)
};

// The 8 collapsed-cone directions used by this teaching model: the 4 axis-aligned
// and 4 diagonal neighbours on the 2-D grid. Real 3-D CCC uses ~48-400 cones on a
// sphere; here 8 in-plane directions are plenty to show the algorithm and still
// give a smooth, isotropic-ish spread. Index c in [0,8) selects (dx,dy).
//
//   IMPORTANT (why these are FUNCTIONS, not a global array): a file-scope
//   `static const int[]` is host-only storage -- nvcc cannot read it from device
//   code, so a kernel indexing it fails to compile ("identifier undefined in
//   device code"). Returning the value from a CCC_HD (host+device) function makes
//   the table a compile-time switch that both worlds inline identically. This is
//   the portable way to share small constant tables across the HD boundary.
CCC_HD inline int ccc_cone_dx(int c) {
    const int dx[8] = { +1, -1,  0,  0, +1, +1, -1, -1 };
    return dx[c];
}
CCC_HD inline int ccc_cone_dy(int c) {
    const int dy[8] = {  0,  0, +1, -1, +1, -1, +1, -1 };
    return dy[c];
}

// Physical step length (cm) for cone c: axis cones step one voxel edge; diagonal
// cones step sqrt(2) voxel edges. Computing it here (not inline at the call site)
// keeps the CPU and GPU using the identical constant.
CCC_HD inline double ccc_step_cm(int c, double voxel_cm) {
    const bool diagonal = (ccc_cone_dx(c) != 0) && (ccc_cone_dy(c) != 0);
    return diagonal ? voxel_cm * 1.4142135623730951 /* sqrt(2) */ : voxel_cm;
}

// ---------------------------------------------------------------------------
// terma_at: the TERMA released in one voxel, given the radiological depth of that
//   voxel's CENTER measured from the top surface along the beam.
//     rad_depth_g_per_cm2 = integral( rho dl )  from surface to voxel center
//                         = (sum of rho over voxels above) * voxel_cm   [g/cm^2]
//   Beer-Lambert attenuation of the primary fluence then TERMA:
//     Psi   = psi0 * exp( -(mu/rho) * rad_depth )
//     TERMA = (mu/rho) * Psi
//   Units are arbitrary/relative (this is a teaching model, not a calibrated
//   dose engine) -- what matters is the SHAPE (exponential falloff, heterogeneity)
//   and that CPU and GPU agree.
// ---------------------------------------------------------------------------
CCC_HD inline double terma_at(const CccParams& P, double rad_depth_g_per_cm2) {
    const double psi = P.psi0 * exp(-P.mu_over_rho * rad_depth_g_per_cm2);
    return P.mu_over_rho * psi;
}

// ---------------------------------------------------------------------------
// cone_transmit / cone_deposit: the two coefficients of the collapsed-cone 1-D
//   recurrence for a single step of radiological length d_rad (= step_cm * rho).
//
//   The analytic point-kernel along a cone ray is  h(r) = a * exp(-a r)  (it
//   integrates to 1 over r in [0,inf), i.e. it conserves the energy carried by
//   this cone). Marching the ray one step of radiological length d_rad, the dose
//   already travelling in the ray is ATTENUATED by exp(-a d_rad) [it keeps going
//   but some is deposited], and the TERMA released in the current voxel INJECTS
//   the fraction (1 - exp(-a d_rad)) of its cone-share into the travelling dose.
//   The recurrence carried by cone_sweep (below) is:
//       carry_out = carry_in * transmit + T_here * cone_weight
//       dose_here += carry_in * (1 - transmit) * ... (deposited fraction)
//   We fold the standard "deposit what the ray gives up this step" bookkeeping
//   into cone_sweep; here we just expose the transmit factor and the per-cone
//   TERMA weight so both host and device use identical numbers.
// ---------------------------------------------------------------------------
CCC_HD inline double cone_transmit(const CccParams& P, double step_cm, double rho) {
    const double d_rad = step_cm * rho;          // radiological step length (g/cm^2-ish)
    return exp(-P.kernel_a * d_rad);             // fraction of ray dose that survives the step
}

// Each of the n_cones cones carries an equal 1/n_cones share of the released
// TERMA (isotropic split -- real kernels weight cones anisotropically; see THEORY).
CCC_HD inline double cone_weight(const CccParams& P) {
    return 1.0 / static_cast<double>(P.n_cones);
}

// ---------------------------------------------------------------------------
// dose_to_units / units_to_dose: fixed-point quantization for DETERMINISTIC
//   atomic accumulation (PATTERNS.md §3). We multiply the floating dose
//   contribution by dose_scale and round to a signed 64-bit integer; many cones
//   atomicAdd these integers into the dose grid, and because integer addition
//   commutes the total is order-independent -> GPU == CPU exactly, every run.
//   Rounding (not truncation) keeps the quantization unbiased.
// ---------------------------------------------------------------------------
CCC_HD inline long long dose_to_units(const CccParams& P, double dose_contrib) {
    // llround rounds half-away-from-zero identically on host and device.
    return static_cast<long long>(llround(dose_contrib * P.dose_scale));
}

CCC_HD inline double units_to_dose(const CccParams& P, long long units) {
    return static_cast<double>(units) / P.dose_scale;
}
