// ===========================================================================
// src/reference_cpu.h  --  Ensemble config, network builder, CPU reference
// ---------------------------------------------------------------------------
// Project 6.11 : Stochastic (Gillespie) Biochemical Simulation
//
// The "job" here is: a ReactionNetwork (built from a small text file) plus a
// count of how many independent SSA trajectories to run. This header declares:
//   * EnsembleConfig       -- the network + trajectory count + seed.
//   * load_config()        -- parse the sample file into an EnsembleConfig.
//   * build_gene_network() -- turn the parsed scalars into a ReactionNetwork.
//   * simulate_cpu()       -- run every trajectory serially (the GPU baseline).
// The per-trajectory physics/RNG live in ssa.h (shared host+device). This file
// is plain C++ (compiled by cl.exe); kernels.cu #includes it too, reusing the
// EnsembleConfig / network builder so CPU and GPU build the *same* network.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "ssa.h"   // ReactionNetwork, TrajectoryResult, simulate_trajectory, SSA_MAX_*

// ---------------------------------------------------------------------------
// EnsembleConfig: everything needed to define the run.
//   THE SAMPLE MODEL (canonical "constitutive gene expression", birth-death of
//   a single species -- mRNA -- with an ANALYTICALLY KNOWN answer):
//       R1:   0    -> M     rate k_prod         (transcription, zeroth order)
//       R2:   M    -> 0     rate k_deg * x_M    (degradation, first order)
//   This immigration-death process has a Poisson stationary distribution with
//   mean = k_prod / k_deg and variance = k_prod / k_deg. We exploit that: the
//   ensemble mean of the time-averaged count must land near k_prod/k_deg, which
//   validates the SCIENCE, not just CPU==GPU agreement (PATTERNS.md section 4).
// ---------------------------------------------------------------------------
struct EnsembleConfig {
    double   k_prod    = 0.0;   // transcription rate   (molecules / time)
    double   k_deg     = 0.0;   // mRNA degradation rate (1 / time)
    uint64_t m0        = 0;     // initial mRNA count
    double   t_end     = 0.0;   // simulation horizon (time units)
    int      n_traj    = 0;     // number of independent trajectories (ensemble size)
    uint64_t base_seed = 0;     // RNG base seed (reproducibility)
};

// The analytic stationary mean of the birth-death model: k_prod / k_deg.
//   Exposed so main.cu can print the theoretical target next to the measured
//   ensemble mean -- the headline "did we recover the physics?" comparison.
inline double analytic_mean(const EnsembleConfig& c) {
    return (c.k_deg > 0.0) ? c.k_prod / c.k_deg : 0.0;
}

// Load an EnsembleConfig from the text format (see data/README.md):
//   "k_prod k_deg m0 t_end n_traj base_seed"
// Throws std::runtime_error on a missing/malformed file so demos fail loudly.
EnsembleConfig load_config(const std::string& path);

// Build the concrete ReactionNetwork (stoichiometry + propensity metadata) that
// both the CPU reference and the GPU kernel simulate. Declared here, defined in
// reference_cpu.cpp, and called from BOTH sides so the network is identical.
ReactionNetwork build_gene_network(const EnsembleConfig& c);

// CPU reference: simulate every trajectory serially into `results` (sized
// n_traj). The trusted baseline the GPU ensemble is verified against -- same
// simulate_trajectory() + same seeds -> bit-identical results.
void simulate_cpu(const EnsembleConfig& c, std::vector<TrajectoryResult>& results);
