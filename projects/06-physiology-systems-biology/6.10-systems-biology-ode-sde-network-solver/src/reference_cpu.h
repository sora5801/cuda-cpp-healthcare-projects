// ===========================================================================
// src/reference_cpu.h  --  Ensemble config + (idx -> params) map + CPU reference
// ---------------------------------------------------------------------------
// Project 6.10 : Systems-Biology ODE/SDE Network Solver
//
// The ensemble is a 2-D parameter SWEEP over the repressilator's two most
// interesting knobs: na values of alpha (max transcription rate) x nn values of
// n (Hill cooperativity) = na*nn independent ODE solves. The config, the flat
// (idx -> alpha,n) mapping, and the shared initial condition live here (as
// __host__ __device__ where the kernel reuses them); the ODE/RK4 itself is in
// grn.h. This header is pure C++ so kernels.cu can #include it safely (it never
// contains a __global__ declaration -- that lives in kernels.cuh).
//
// READ THIS AFTER: grn.h; BEFORE: reference_cpu.cpp, kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "grn.h"   // GRN_HD, GrnParams, MemberResult, integrate_member, STATE_DIM

// One ensemble job: fixed model/integration settings + the two swept ranges.
struct EnsembleConfig {
    // Fixed (shared by every member) repressilator parameters.
    double alpha0 = 0.0;   // leaky basal transcription (promoter floor)
    double beta   = 0.0;   // protein-decay / mRNA-decay ratio
    // Fixed integration settings.
    double dt    = 0.0;    // RK4 timestep (in mRNA-lifetime units)
    int    steps = 0;      // number of timesteps (run length = steps*dt)
    // Fixed initial condition (length STATE_DIM = [m0 m1 m2 p0 p1 p2]). A small
    // asymmetric seed is used so the ring does not sit at the symmetric fixed
    // point where a perfectly uniform start would keep it artificially still.
    double s0[STATE_DIM] = {0.0};
    // Sweep grid: na alpha-values x nn n-values.
    int    na = 0, nn = 0;
    double alpha_lo = 0.0, alpha_hi = 0.0;   // max-transcription-rate range
    double n_lo = 0.0,     n_hi = 0.0;       // Hill-coefficient range
};

// Number of ensemble members (one ODE solve each).
GRN_HD inline int ensemble_size(const EnsembleConfig& c) { return c.na * c.nn; }

// Map a flat member index to its (alpha, n) on the sweep grid.
//   idx = a*nn + b  ->  alpha from row a in [alpha_lo,alpha_hi],
//                       n     from col b in [n_lo,n_hi]   (linear, inclusive).
// Endpoint-inclusive linear spacing keeps the corners of the sweep exact so the
// printed sample members land on round parameter values. The fixed fields
// (alpha0, beta) are copied in so a member's GrnParams is fully populated.
GRN_HD inline void member_params(const EnsembleConfig& c, int idx, GrnParams& pr) {
    const int a = idx / c.nn;    // alpha index (row)
    const int b = idx % c.nn;    // n index (col)
    pr.alpha  = (c.na > 1) ? c.alpha_lo + (c.alpha_hi - c.alpha_lo) * a / (c.na - 1) : c.alpha_lo;
    pr.n      = (c.nn > 1) ? c.n_lo     + (c.n_hi     - c.n_lo)     * b / (c.nn - 1) : c.n_lo;
    pr.alpha0 = c.alpha0;
    pr.beta   = c.beta;
}

// Load an EnsembleConfig from the text format (see data/README.md):
//   "alpha0 beta dt steps na nn alpha_lo alpha_hi n_lo n_hi m0 m1 m2 p0 p1 p2"
// Throws std::runtime_error on a missing file or malformed / invalid parameters
// so demos fail loudly rather than running on garbage.
EnsembleConfig load_ensemble(const std::string& path);

// CPU reference: integrate every member serially into results (sized na*nn).
// This is the trusted baseline the GPU ensemble is checked against; because both
// call integrate_member() from grn.h, they perform identical arithmetic.
void integrate_cpu(const EnsembleConfig& c, std::vector<MemberResult>& results);
