// ===========================================================================
// src/lung_physics.h  --  Shared (host + device) aerosol-deposition physics
// ---------------------------------------------------------------------------
// Project 6.5 : Respiratory / Lung Airflow & Particle Deposition
//
// WHY THIS HEADER IS SHARED
//   The whole point of the verification here is that the CPU reference and the
//   GPU kernel track the *identical* particle histories, so their per-generation
//   deposition tallies must match EXACTLY. That only works if both sides use the
//   same RNG and the same deposition physics -- so both live here, in ONE header
//   included by reference_cpu.cpp (host compiler) AND kernels.cu / main.cu
//   (nvcc). This is the "shared __host__ __device__ core" idiom of
//   docs/PATTERNS.md section 2, exactly as used in flagships 5.01, 6.04, 9.02.
//
//   The LUNG_HD macro expands to `__host__ __device__` under nvcc and to nothing
//   under the plain host compiler, so the same inline functions compile in both
//   worlds. Keep this header free of `__global__` and CUDA-only types so cl.exe
//   can include it too.
//
// THE SIMPLIFIED PHYSICS (a deliberately REDUCED teaching model -- THEORY.md has
// the full CFD picture and cites the ICRP / MPPD deposition models this follows)
//   The conducting airways are idealized as a SYMMETRIC bifurcating tree (the
//   Weibel "A" model). Generation g (0 = trachea, deeper = larger g) is a
//   straight round tube of radius r_g and length L_g through which air flows at
//   mean velocity U_g. A single inhaled aerosol particle enters the trachea and
//   is followed tube-by-tube. In EACH generation three classical mechanisms can
//   remove ("deposit") the particle onto the airway wall:
//
//     1. INERTIAL IMPACTION -- a heavy/fast particle cannot follow the ~90-deg
//        turn at a bifurcation and hits the wall. Governed by the Stokes number
//        Stk = rho_p d_p^2 U C_c / (18 mu r).  Deposition eff eta_imp = f(Stk).
//     2. GRAVITATIONAL SEDIMENTATION -- the particle settles at its terminal
//        velocity v_s = rho_p d_p^2 g C_c / (18 mu) during the residence time
//        t_res = L/U it spends in the tube.  eta_sed grows with t_res / r.
//     3. BROWNIAN DIFFUSION -- tiny (sub-micron) particles random-walk to the
//        wall. Diffusion coefficient D = k_B T C_c / (3 pi mu d_p); eta_diff
//        grows with the diffusion parameter D t_res / r^2.
//
//   The three per-generation efficiencies are combined into a single survival
//   product: P(pass generation g) = (1-eta_imp)(1-eta_sed)(1-eta_diff). We draw
//   ONE uniform random number xi in [0,1); if xi >= P the particle deposits in
//   generation g (tally++ and stop), otherwise it survives to generation g+1.
//   A particle that survives all N generations is "exhaled" (tallied separately).
//
//   Everything above is DOUBLE-PRECISION float math and is bit-identical on host
//   and device (no fast-math, no atomics in the trajectory). The only thing we
//   atomically accumulate is an INTEGER count per generation -- integer adds
//   commute, so the GPU tally is deterministic and equals the CPU tally exactly
//   (docs/PATTERNS.md section 3). See ../THEORY.md for the derivations.
// ===========================================================================
#pragma once

#include <cstdint>
#include <cmath>

#ifdef __CUDACC__
#define LUNG_HD __host__ __device__
#else
#define LUNG_HD
#endif

// ---------------------------------------------------------------------------
// Physical + numerical constants (SI units unless noted). Declared as inline
// constexpr so both translation units see the SAME values with no ODR trouble.
// ---------------------------------------------------------------------------
namespace lung {

constexpr double PI       = 3.14159265358979323846;
constexpr double K_B      = 1.380649e-23;   // Boltzmann constant           [J/K]
constexpr double G_ACCEL  = 9.80665;        // standard gravity             [m/s^2]
constexpr double MU_AIR   = 1.81e-5;        // dynamic viscosity of air     [Pa*s]
constexpr double MFP_AIR  = 6.8e-8;         // air mean free path (~68 nm)  [m]
constexpr double T_BODY   = 310.15;         // body temperature (37 C)      [K]

// Maximum airway generations we support (Weibel model runs 0..23; the conducting
// zone is 0..16). Fixed size keeps per-thread state in registers/local arrays.
constexpr int MAX_GEN = 32;

// ---------------------------------------------------------------------------
// Cunningham slip correction C_c(d_p): sub-micron particles are comparable in
// size to the air mean free path, so the no-slip continuum drag law over-
// predicts drag. C_c >= 1 corrects for this. Knudsen number Kn = 2*lambda/d_p.
//   (Standard Davies 1945 coefficients; identical host/device.)
// ---------------------------------------------------------------------------
LUNG_HD inline double cunningham(double d_p) {
    const double Kn = 2.0 * MFP_AIR / d_p;   // Knudsen number (dimensionless)
    return 1.0 + Kn * (1.257 + 0.400 * exp(-1.10 / Kn));
}

// ---------------------------------------------------------------------------
// A particle's fixed properties, computed once from its diameter and density.
//   tau  : particle relaxation time  = rho_p d_p^2 C_c / (18 mu)   [s]
//          (how quickly the particle catches up to the flow; drives impaction)
//   v_s  : gravitational settling velocity = tau * g               [m/s]
//   D    : Brownian diffusion coefficient  = k_B T C_c / (3 pi mu d_p) [m^2/s]
// ---------------------------------------------------------------------------
struct Particle {
    double d_p;    // aerodynamic diameter                          [m]
    double rho_p;  // particle mass density                         [kg/m^3]
    double tau;    // relaxation time                               [s]
    double v_s;    // settling velocity                             [m/s]
    double D;      // diffusion coefficient                         [m^2/s]
};

LUNG_HD inline Particle make_particle(double d_p, double rho_p) {
    Particle p;
    p.d_p   = d_p;
    p.rho_p = rho_p;
    const double Cc = cunningham(d_p);
    p.tau = rho_p * d_p * d_p * Cc / (18.0 * MU_AIR);
    p.v_s = p.tau * G_ACCEL;
    p.D   = K_B * T_BODY * Cc / (3.0 * PI * MU_AIR * d_p);
    return p;
}

// ---------------------------------------------------------------------------
// Airway description: the symmetric bifurcating tree. Each generation g has a
// tube radius r[g], length L[g], and mean axial air velocity U[g]. These come
// from a Weibel-style geometry scaled by the tidal flow rate (see build_airway
// in reference_cpu.cpp). n_gen generations are used (conducting zone).
// ---------------------------------------------------------------------------
struct Airway {
    int    n_gen;             // number of generations actually used (<= MAX_GEN)
    double r[MAX_GEN];        // airway radius per generation                 [m]
    double L[MAX_GEN];        // airway length per generation                 [m]
    double U[MAX_GEN];        // mean axial air velocity per generation        [m/s]
};

// ---------------------------------------------------------------------------
// Per-generation deposition efficiencies. Each returns a probability in [0,1].
// These are the standard semi-empirical correlations used in ICRP-66 / MPPD
// (see THEORY.md "real world"); we keep the classic closed forms so the learner
// can trace every term back to the physics above.
// ---------------------------------------------------------------------------

// Inertial impaction at the bifurcation.  Stokes number Stk = tau*U / r.
// eta_imp = 1 - exp(-k * Stk) is a simple, monotone, well-behaved form; the
// constant k folds in the branching-angle geometry (we use k = 2 for a ~35-deg
// half-angle bifurcation, a common textbook value).
LUNG_HD inline double eta_impaction(const Particle& p, double U, double r) {
    const double Stk = p.tau * U / r;             // dimensionless Stokes number
    return 1.0 - exp(-2.0 * Stk);
}

// Gravitational sedimentation in a horizontal round tube. During residence time
// t_res = L/U the particle settles a distance v_s * t_res; the fraction of the
// circular cross-section swept scales as (v_s * t_res)/(2 r). We clamp to [0,1].
LUNG_HD inline double eta_sedimentation(const Particle& p, double U, double r, double L) {
    const double t_res = L / U;                   // residence time in the tube [s]
    double eff = (p.v_s * t_res) / (2.0 * r);     // swept-fraction estimate
    if (eff < 0.0) eff = 0.0;
    if (eff > 1.0) eff = 1.0;
    return eff;
}

// Brownian diffusion to the wall (Ingham 1975 tube-diffusion form). The single
// dimensionless group is Delta = D * t_res / r^2. The leading term of Ingham's
// series, eta_diff = 1 - exp(-5.784 * Delta), captures the sub-micron rise.
LUNG_HD inline double eta_diffusion(const Particle& p, double U, double r, double L) {
    const double t_res = L / U;
    const double Delta = p.D * t_res / (r * r);   // dimensionless diffusion group
    return 1.0 - exp(-5.784 * Delta);
}

// ---------------------------------------------------------------------------
// RNG: a splitmix64 counter-based stream, identical on host and device. Each
// particle seeds an independent, reproducible stream from its index, so the CPU
// and GPU replay the exact same random draws -> the tallies match bit-for-bit.
// (Production Lagrangian codes use cuRAND; we use a shared reproducible RNG on
// purpose, for deterministic verification -- docs/PATTERNS.md section 3.)
// ---------------------------------------------------------------------------
struct Rng { uint64_t state; };

LUNG_HD inline uint64_t splitmix64(uint64_t& x) {
    x += 0x9E3779B97F4A7C15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Seed an independent stream for particle `idx` from a base seed.
LUNG_HD inline Rng rng_seed(uint64_t base, uint64_t idx) {
    Rng r;
    r.state = base ^ (idx * 0x9E3779B97F4A7C15ULL + 0xD1B54A32D192ED03ULL);
    splitmix64(r.state);   // warm up the state once
    return r;
}

// Uniform double in [0,1) from 53 random bits (identical math host/device).
LUNG_HD inline double rng_uniform(Rng& r) {
    uint64_t z = splitmix64(r.state);
    return (z >> 11) * (1.0 / 9007199254740992.0);   // multiply by 2^-53
}

// ---------------------------------------------------------------------------
// track_particle: follow ONE particle through the airway tree and return the
// generation in which it deposits, or `airway.n_gen` if it is exhaled.
//
//   PARAMETERS
//     p     : the particle's fixed properties (diameter, density, tau, v_s, D)
//     aw    : the airway geometry (radii, lengths, velocities per generation)
//     rng   : this particle's private RNG stream (advanced as we draw)
//   RETURNS
//     g in [0, n_gen-1]  -> deposited in generation g
//     n_gen              -> survived all generations (exhaled)
//
//   The loop is pure double-precision + one integer comparison per generation,
//   so it is bit-identical on host and device. The CALLER converts the returned
//   generation into an integer tally (plain ++ on CPU, atomicAdd on GPU) -- that
//   split is what lets identical physics feed two accumulation strategies while
//   staying exactly equal (integer counts commute).
// ---------------------------------------------------------------------------
LUNG_HD inline int track_particle(const Particle& p, const Airway& aw, Rng& rng) {
    for (int g = 0; g < aw.n_gen; ++g) {
        // Combine the three independent removal mechanisms into a survival
        // probability. (1 - eta) is the chance the particle SURVIVES that
        // mechanism; multiplying assumes the mechanisms act independently over
        // the tube -- the standard "deposition-probability" assumption.
        const double e_imp = eta_impaction   (p, aw.U[g], aw.r[g]);
        const double e_sed = eta_sedimentation(p, aw.U[g], aw.r[g], aw.L[g]);
        const double e_dif = eta_diffusion   (p, aw.U[g], aw.r[g], aw.L[g]);
        const double survive = (1.0 - e_imp) * (1.0 - e_sed) * (1.0 - e_dif);

        // One random draw decides deposition in THIS generation.
        const double xi = rng_uniform(rng);
        if (xi >= survive) return g;   // deposited here -> stop
    }
    return aw.n_gen;                   // survived every generation -> exhaled
}

}  // namespace lung
