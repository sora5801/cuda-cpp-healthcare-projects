// ===========================================================================
// src/reference_cpu.h  --  LBM parameters + CPU reference solver
// ---------------------------------------------------------------------------
// Project 6.04 : Lattice-Boltzmann Blood/Airflow Solver
//
// Pure C++ (no CUDA). kernels.cu reuses LbmParams. The actual per-node physics
// is in the shared lbm_d2q9.h so CPU and GPU are byte-for-byte identical.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "lbm_d2q9.h"   // shared update (host+device)

// A simulation job: a periodic channel (walls top/bottom) driven by a body force.
struct LbmParams {
    int nx = 0, ny = 0;   // lattice size (x = flow direction, y = across channel)
    int steps = 0;        // number of collide+stream iterations
    double tau = 0.0;     // BGK relaxation time -> kinematic viscosity nu=(tau-0.5)/3
    double gx = 0.0;      // body force per unit mass in +x (the pressure gradient)
};

// Load LbmParams from the one-line text format (data/README.md):
//   "nx ny steps tau gx"
LbmParams load_lbm(const std::string& path);

// CPU reference: initialize populations at rest equilibrium, run `steps`
// collide+stream iterations (ping-ponging two buffers), and return the FINAL
// population field `f` (size 9*nx*ny). The trusted baseline for the GPU.
void lbm_cpu(const LbmParams& p, std::vector<double>& f_final);

// Fill `ux` (size nx*ny) with the macroscopic x-velocity of every node, from a
// population field. Shared by both paths so the comparison is apples-to-apples.
void velocity_field(const LbmParams& p, const std::vector<double>& f, std::vector<double>& ux);
