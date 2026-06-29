// ===========================================================================
// src/reference_cpu.h  --  Periodic charge system + CPU Ewald references
// ---------------------------------------------------------------------------
// Project 1.2 : Particle-Mesh Ewald Electrostatics
//
// WHAT'S HERE
//   * System          -- N point charges in a CUBIC periodic box (the input).
//   * load_system     -- read the tiny text sample (data/README.md format).
//   * PmeParams       -- grid size, B-spline order, Ewald beta, real cutoff.
//   * The CPU REFERENCES the GPU is checked against:
//       - ewald_recip_direct_cpu : the textbook reciprocal-space k-vector sum.
//             This is the SLOW-but-obviously-correct gold standard for E_recip.
//       - pme_recip_cpu          : the SAME spread->DFT->convolve->energy pipeline
//             the GPU runs, on the host, using the shared pme.h math. Verifying
//             GPU == pme_recip_cpu isolates "did cuFFT + the kernels reproduce
//             the host pipeline" from "is SPME a good approximation to Ewald".
//       - ewald_real_cpu / ewald_self : the short-range and self terms, so we can
//             assemble the FULL Ewald energy and check it is INVARIANT to beta.
//
//   Pure C++ header (no CUDA): kernels.cu reuses System + PmeParams.
//
// READ THIS BEFORE: reference_cpu.cpp, main.cu.  READ pme.h FIRST (shared math).
// ===========================================================================
#pragma once

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// System: N point charges in a cubic periodic box of side `box` (one length
// unit, see pme.h). Coordinates are wrapped into [0, box). The net charge is
// (approximately) zero -- a requirement for the periodic Coulomb sum to be
// well-defined; the loader checks it and the self/neutralizing terms assume it.
// ---------------------------------------------------------------------------
struct System {
    int    n   = 0;                 // number of charges
    double box = 0.0;               // cubic box side length (same unit as coords)
    std::vector<double> x, y, z;    // [n] Cartesian coordinates, each in [0, box)
    std::vector<double> q;          // [n] charges in elementary units (e)
};

// ---------------------------------------------------------------------------
// PmeParams: the knobs of the method. `K` is the FFT grid points per axis (the
// grid is K x K x K). `order` is the B-spline order (= PME_ORDER from pme.h).
// `beta` is the Ewald splitting parameter (1/length). `rcut` is the real-space
// cutoff used by ewald_real_cpu (and must be <= box/2 for minimum image).
// ---------------------------------------------------------------------------
struct PmeParams {
    int    K     = 32;
    int    order = 4;
    double beta  = 0.0;
    double rcut  = 0.0;
};

// Load a System from the text format documented in data/README.md:
//   header:  "<n> <box>"   then n rows of "x y z q".
System load_system(const std::string& path);

// Choose sensible PME parameters for a system: grid size K (>= a few per length
// unit, rounded up to an FFT-friendly size), beta from a target real-space
// accuracy at the cutoff, and rcut = box/2 (the largest minimum-image cutoff).
// Deterministic given the system, so CPU and GPU use identical parameters.
PmeParams choose_params(const System& s);

// --- The reciprocal-space references ---------------------------------------

// Gold standard: the direct Ewald reciprocal sum over integer wavevectors
//   E_recip = (2*pi / V) * sum_{m != 0} exp(-pi^2 |m|^2 / beta^2) / |m|^2 * |S(m)|^2
// with structure factor S(m) = sum_j q_j exp(2*pi*i m . r_j / box). O(N * Kmax^3)
// -- slow, but transparently correct. The truth E_recip is measured against.
double ewald_recip_direct_cpu(const System& s, const PmeParams& p);

// The SPME pipeline on the host (the exact twin of the GPU path): build the
// B-spline charge grid, forward DFT, multiply by the Ewald influence function,
// and sum the reciprocal energy. Uses pme.h so it matches the GPU bit-closely.
// Returns E_recip.
double pme_recip_cpu(const System& s, const PmeParams& p);

// --- The other Ewald terms (so we can form the TOTAL energy) ---------------

// Short-range real-space sum with the minimum-image convention and erfc damping:
//   E_real = sum_{i<j, r_ij<rcut} q_i q_j * erfc(beta r_ij) / r_ij
double ewald_real_cpu(const System& s, const PmeParams& p);

// Self-energy correction (subtracted): each Gaussian interacts with itself.
//   E_self = (beta / sqrt(pi)) * sum_i q_i^2
double ewald_self(const System& s, const PmeParams& p);

// Convenience: assemble E_total = E_real + E_recip - E_self using the DIRECT
// reciprocal sum (the physically exact Ewald energy). Used for the beta-
// invariance physics check in main.cu.
double ewald_total_direct_cpu(const System& s, const PmeParams& p);

// Build the SPME reciprocal-space "influence function" array B(m)*C(m) for the
// half-complex grid (size K*K*(K/2+1)), shared by the host pipeline and uploaded
// to the GPU so both convolve with identical coefficients. Layout matches the
// cuFFT R2C output (see kernels.cu). Exposed so main.cu can hand it to the GPU.
void build_influence(const System& s, const PmeParams& p, std::vector<double>& influence);

// Build the B-spline charge grid as fixed-point integers (the exact accumulator
// the GPU atomics build), then return it as a real grid of size K*K*K in
// row-major (ix slowest ... iz fastest) order. Shared so the host pipeline and
// the GPU spreading kernel can be compared cell-by-cell if desired.
void spread_charges_cpu(const System& s, const PmeParams& p, std::vector<double>& grid);
