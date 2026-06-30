// ===========================================================================
// src/reference_cpu.h  --  System config + CPU reference SCF solver interface
// ---------------------------------------------------------------------------
// Project 2.27 : Polarizable Water Model GPU Dynamics
//
// This header declares:
//   * PolarSystem  -- the loaded cluster of polarizable sites + solver settings.
//   * load_system  -- parse the tiny text format in data/sample/.
//   * compute_permanent_field_cpu / solve_dipoles_cpu  -- the trusted serial
//     baseline the GPU is checked against (it runs the SAME Jacobi iteration on
//     the SAME shared physics in polar.h, so the two agree to round-off).
//   * SolveResult  -- the converged dipoles + diagnostics returned to main.cu.
//
// The per-site electrostatics math lives in polar.h (shared host+device). This
// file is PURE C++ (no CUDA), so kernels.cu can reuse PolarSystem/SolveResult
// while reference_cpu.cpp is compiled by the plain host compiler.
//
// READ AFTER: polar.h.  READ BEFORE: reference_cpu.cpp, kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "polar.h"   // Site, Vec3, POLAR_HD, field/energy helpers

// ---------------------------------------------------------------------------
// PolarSystem: everything needed to define and solve one induced-dipole problem.
//   sites        : the N polarizable/charged sites (geometry + q + alpha).
//   Eext         : a uniform EXTERNAL field applied to every site (e/A^2). Lets
//                  us pose the analytic check "isolated dipole in a field gives
//                  mu = alpha*E" and to drive the cluster. Usually zero.
//   a_thole      : Thole exponential screening parameter (dimensionless, ~0.39
//                  in AMOEBA; we expose it so learners can see damping matter).
//   max_iters    : Jacobi sweep cap (the SCF loop stops early once converged).
//   tol          : convergence threshold on max |dmu| between sweeps (e*A).
// ---------------------------------------------------------------------------
struct PolarSystem {
    std::vector<Site> sites;
    Vec3   Eext{0.0, 0.0, 0.0};
    double a_thole = 0.39;
    int    max_iters = 200;
    double tol = 1.0e-9;
};

// Number of sites (convenience).
inline int num_sites(const PolarSystem& s) { return static_cast<int>(s.sites.size()); }

// ---------------------------------------------------------------------------
// SolveResult: the converged induced dipoles plus solver diagnostics. main.cu
// compares the CPU and GPU versions of this struct's dipoles + energy.
//   mu            : converged induced dipole per site (e*A), length N.
//   iters         : number of Jacobi sweeps actually taken.
//   final_dmu     : the max per-component dipole change at the last sweep (the
//                   residual; <= tol means converged).
//   U_pol         : total polarization energy in internal units (e^2/A).
//   U_pol_kcal    : the same energy in kcal/mol (chemically meaningful).
// ---------------------------------------------------------------------------
struct SolveResult {
    std::vector<Vec3> mu;
    int    iters = 0;
    double final_dmu = 0.0;
    double U_pol = 0.0;
    double U_pol_kcal = 0.0;
};

// Parse a PolarSystem from the whitespace text format documented in
// data/README.md. Throws std::runtime_error on a malformed/missing file so the
// demo fails loudly rather than running on garbage.
PolarSystem load_system(const std::string& path);

// Compute the PERMANENT field E^perm_i at every site (from the static charges of
// all other sites, plus the uniform external field). This is independent of the
// induced dipoles, so it is computed ONCE and reused every SCF sweep. Output
// `Eperm` is resized to N. Shared structurally with the GPU's permanent-field
// kernel; both call field_perm_pair() from polar.h.
void compute_permanent_field_cpu(const PolarSystem& sys, std::vector<Vec3>& Eperm);

// The CPU reference SCF solver: Jacobi-iterate mu_i = alpha_i * E_i until the
// dipoles converge (or max_iters). Fills a SolveResult. This is the ground truth
// the GPU result is verified against (same arithmetic -> agreement to round-off).
SolveResult solve_dipoles_cpu(const PolarSystem& sys);
