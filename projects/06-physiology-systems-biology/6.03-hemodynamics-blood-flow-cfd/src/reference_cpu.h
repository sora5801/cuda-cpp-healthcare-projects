// ===========================================================================
// src/reference_cpu.h  --  Channel parameters + CPU reference NSE solver
// ---------------------------------------------------------------------------
// Project 6.3 : Hemodynamics / Blood-Flow CFD   (reduced-scope teaching version)
//
// Pure C++ (NO CUDA) so this header is safe to include from both the plain host
// compiler (reference_cpu.cpp) and from nvcc (kernels.cu reuses ChannelParams).
// The per-cell physics lives in the shared nse_channel.h so the CPU and GPU
// solvers run byte-for-byte identical math (PATTERNS.md §2).
//
// READ THIS AFTER: nse_channel.h (the physics). READ THIS BEFORE: kernels.cuh.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "nse_channel.h"   // idx, predictor/pressure/corrector cell physics

// ---------------------------------------------------------------------------
// ChannelParams: a complete 2-D incompressible-NSE channel-flow job.
//   Loaded from the one-line sample file (see data/README.md). Units are kept
//   consistent (dimensionless "lattice" units in the demo) so results are clean
//   round numbers; THEORY.md shows how to map to SI (cm, s, Pa).
// ---------------------------------------------------------------------------
struct ChannelParams {
    int    nx = 0, ny = 0;   // grid size: x=streamwise, y=across the channel
    int    steps = 0;        // number of time steps to advance
    int    p_iters = 0;      // Jacobi iterations per pressure Poisson solve
    double h = 0.0;          // uniform cell spacing (same in x and y)
    double dt = 0.0;         // time-step size
    double rho = 0.0;        // fluid density (blood ~ 1060 kg/m^3)
    double gx = 0.0;         // streamwise body force (stand-in for dp/dx)
    // Carreau-Yasuda viscosity parameters. Setting nu0 == nu_inf selects a
    // NEWTONIAN fluid (constant viscosity) -> matches analytic Poiseuille, which
    // is how we verify correctness. nu0 != nu_inf enables shear thinning.
    double nu0 = 0.0;        // zero-shear kinematic viscosity
    double nu_inf = 0.0;     // infinite-shear kinematic viscosity
    double lambda = 0.0;     // Carreau-Yasuda relaxation time
    double n_cy = 1.0;       // power-law index (<1 => shear thinning)
    double a_cy = 2.0;       // Yasuda transition exponent
};

// Load ChannelParams from the whitespace-separated text format documented in
// data/README.md. Throws std::runtime_error on a missing/malformed file.
ChannelParams load_channel(const std::string& path);

// CPU reference solver: run `steps` fractional-step (Chorin projection) updates
//   starting from rest (u=v=0, p=0) and return the final velocity fields `u` and
//   `v` (each size nx*ny, row-major). This is the trusted baseline the GPU is
//   checked against. Implementation loops the shared per-cell functions from
//   nse_channel.h over every cell, exactly as the GPU launches one thread per
//   cell.
void nse_cpu(const ChannelParams& p,
             std::vector<double>& u_final,
             std::vector<double>& v_final);

// Compute the effective (constant) kinematic viscosity used by the analytic
// check. For a Newtonian job this is simply nu0; documented in THEORY.md.
double effective_nu(const ChannelParams& p);

// Analytic steady Poiseuille centreline velocity for a Newtonian channel:
//     u_max = gx * (H/2)^2 / (2*nu),   H = (ny-1)*h  (wall-to-wall height)
// Used as a SECOND, science-level check (PATTERNS.md §4): the simulation should
// converge toward this known value, not merely agree CPU-vs-GPU.
double poiseuille_umax(const ChannelParams& p);
