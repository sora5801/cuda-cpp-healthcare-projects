// ===========================================================================
// src/reference_cpu.cpp  --  Config loader, network builder, serial SSA driver
// ---------------------------------------------------------------------------
// Project 6.11 : Stochastic (Gillespie) Biochemical Simulation
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU result is checked against. It runs the SAME
//   simulate_trajectory() (ssa.h) as the kernel, just in a plain serial loop, so
//   when the GPU and CPU agree we believe the GPU. Compiled by the host C++
//   compiler only (no CUDA here). See reference_cpu.h and ssa.h.
//
// READ THIS AFTER: reference_cpu.h, ssa.h. Compare against kernels.cu (GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// load_config: parse the whitespace-separated sample file into an
//   EnsembleConfig. Format (one line, see data/README.md):
//       k_prod  k_deg  m0  t_end  n_traj  base_seed
//   We validate ranges so a corrupt file fails loudly instead of silently
//   simulating nonsense (e.g. k_deg <= 0 would make the mean diverge).
// ---------------------------------------------------------------------------
EnsembleConfig load_config(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open config file: " + path);

    EnsembleConfig c;
    if (!(in >> c.k_prod >> c.k_deg >> c.m0 >> c.t_end >> c.n_traj >> c.base_seed))
        throw std::runtime_error("bad parameters (expected "
            "'k_prod k_deg m0 t_end n_traj base_seed') in " + path);

    if (c.k_prod <= 0.0 || c.k_deg <= 0.0 || c.t_end <= 0.0 || c.n_traj <= 0)
        throw std::runtime_error("invalid config (need k_prod>0, k_deg>0, "
                                 "t_end>0, n_traj>0) in " + path);
    return c;
}

// ---------------------------------------------------------------------------
// build_gene_network: assemble the concrete ReactionNetwork for the birth-death
//   gene-expression model. ONE species (M = mRNA) and TWO reactions:
//       R1:  0 -> M    order 0, rate k_prod            nu = (+1)
//       R2:  M -> 0    order 1 in M, rate k_deg        nu = (-1)
//   Called by BOTH reference_cpu.cpp (here) and kernels.cu, so the two sides
//   simulate byte-for-byte the same network. The struct is POD, so the GPU
//   receives it by value -- no device allocation for the network itself.
// ---------------------------------------------------------------------------
ReactionNetwork build_gene_network(const EnsembleConfig& c) {
    ReactionNetwork net{};          // zero-initialise every slot (nu, indices...)
    net.n_species   = 1;            // species 0 = M (mRNA)
    net.n_reactions = 2;
    net.x0[0]       = c.m0;         // initial mRNA count
    net.t_end       = c.t_end;
    net.base_seed   = c.base_seed;

    // R1: transcription  0 -> M  (a constant "source"): order 0, propensity k_prod.
    net.k[0]         = c.k_prod;
    net.order[0]     = 0;
    net.reactant1[0] = -1;          // no reactant
    net.reactant2[0] = -1;
    net.nu[0][0]     = +1;          // produce one M

    // R2: degradation   M -> 0: first order in M, propensity k_deg * x_M.
    net.k[1]         = c.k_deg;
    net.order[1]     = 1;
    net.reactant1[1] = 0;           // reactant is species 0 (M)
    net.reactant2[1] = -1;
    net.nu[1][0]     = -1;          // consume one M

    return net;
}

// ---------------------------------------------------------------------------
// simulate_cpu: run every trajectory serially. Each trajectory is independent
//   (its own RNG stream keyed by its index), so this loop is the direct serial
//   analogue of "one GPU thread per trajectory" in kernels.cu. Same network,
//   same seeds -> results[i] here == results[i] on the GPU, bit for bit.
// ---------------------------------------------------------------------------
void simulate_cpu(const EnsembleConfig& c, std::vector<TrajectoryResult>& results) {
    const ReactionNetwork net = build_gene_network(c);
    results.assign(static_cast<std::size_t>(c.n_traj), TrajectoryResult{});
    for (int i = 0; i < c.n_traj; ++i) {
        // Trajectory i uses RNG stream i (seeded inside simulate_trajectory from
        // net.base_seed ^ f(i)); this is what makes the ensemble reproducible.
        results[static_cast<std::size_t>(i)] =
            simulate_trajectory(net, static_cast<uint64_t>(i));
    }
}
