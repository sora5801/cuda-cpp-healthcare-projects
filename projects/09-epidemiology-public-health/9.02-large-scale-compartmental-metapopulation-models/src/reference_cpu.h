// ===========================================================================
// src/reference_cpu.h  --  Ensemble config + CPU reference integration
// ---------------------------------------------------------------------------
// Project 9.02 : Large-Scale Compartmental & Metapopulation Models
//
// The ensemble is a 2-D parameter SWEEP: nb values of beta x ng values of gamma
// = nb*ng independent SEIR solves. The config + the (idx -> beta,gamma) mapping
// live here (shared host+device so the kernel reuses them); the actual ODE/RK4
// is in seir.h. Pure C++; kernels.cu reuses EnsembleConfig.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "seir.h"   // SEIR_HD, integrate_member, MemberResult

// One ensemble job: fixed population/IC/integration settings + parameter ranges.
struct EnsembleConfig {
    double N = 0.0;       // total population (constant)
    double I0 = 0.0;      // initial infectious count (S0 = N - I0, E0 = R0 = 0)
    double dt = 0.0;      // RK4 timestep (days)
    int    steps = 0;     // number of timesteps (run length = steps*dt days)
    double sigma = 0.0;   // 1 / latent period (E -> I rate)
    int    nb = 0, ng = 0;          // sweep grid: nb beta values x ng gamma values
    double beta_lo = 0.0, beta_hi = 0.0;     // transmission-rate range
    double gamma_lo = 0.0, gamma_hi = 0.0;   // recovery-rate range
};

// Number of ensemble members.
SEIR_HD inline int ensemble_size(const EnsembleConfig& c) { return c.nb * c.ng; }

// Map a flat member index to its (beta, gamma) on the sweep grid.
//   idx = a*ng + b  ->  beta = beta_lo + a/(nb-1)*range, gamma likewise from b.
SEIR_HD inline void member_params(const EnsembleConfig& c, int idx, double& beta, double& gamma) {
    const int a = idx / c.ng;     // beta index
    const int b = idx % c.ng;     // gamma index
    beta  = (c.nb > 1) ? c.beta_lo  + (c.beta_hi  - c.beta_lo)  * a / (c.nb - 1) : c.beta_lo;
    gamma = (c.ng > 1) ? c.gamma_lo + (c.gamma_hi - c.gamma_lo) * b / (c.ng - 1) : c.gamma_lo;
}

// Load an EnsembleConfig from the text format (data/README.md):
//   "N I0 dt steps sigma nb ng beta_lo beta_hi gamma_lo gamma_hi"
EnsembleConfig load_ensemble(const std::string& path);

// CPU reference: integrate every member serially. results sized to nb*ng. The
// trusted baseline the GPU ensemble is checked against (same RK4 -> same numbers).
void integrate_cpu(const EnsembleConfig& c, std::vector<MemberResult>& results);
