// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial multi-walker metadynamics baseline
// ---------------------------------------------------------------------------
// Project 1.6 : Enhanced Sampling -- Metadynamics & Replica Exchange
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU result is checked against. It is OBVIOUSLY correct
//   -- a single readable loop over walkers, each integrated by the shared
//   metad::run_walker() (the exact same code the GPU kernel calls). When CPU and
//   GPU agree to machine precision, we believe the GPU.
//
//   Compiled by the host C++ compiler only (no CUDA here). The per-walker physics
//   lives in metad.h; this file just orchestrates the ensemble + loads the config.
//
// READ THIS AFTER: metad.h, reference_cpu.h. Compare against kernels.cu (GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// load_config: parse the single-line whitespace-separated configuration.
//   Field order (also documented in data/README.md):
//     A kT mass friction dt steps hill_w hill_sigma deposit_every bias_factor
//     s_lo s_hi nbins n_walkers seed s_start
//   We validate the physically meaningful constraints (positive counts, a grid
//   with >= 2 bins, bias_factor > 1 for well-tempered MetaD) so the demo fails
//   loudly on a bad file rather than producing nonsense.
// ---------------------------------------------------------------------------
MetadConfig load_config(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open config file: " + path);

    MetadConfig c;
    metad::Model& m = c.model;          // alias to fill the embedded model
    if (!(in >> m.A >> m.kT >> m.mass >> m.friction >> m.dt >> m.steps
             >> m.hill_w >> m.hill_sigma >> m.deposit_every >> m.bias_factor
             >> m.s_lo >> m.s_hi >> m.nbins
             >> c.n_walkers >> c.seed >> c.s_start)) {
        throw std::runtime_error(
            "bad parameters (expected 'A kT mass friction dt steps hill_w "
            "hill_sigma deposit_every bias_factor s_lo s_hi nbins n_walkers "
            "seed s_start') in " + path);
    }

    // --- Sanity checks: reject configurations that make no physical sense. ---
    if (m.steps <= 0 || m.dt <= 0.0 || m.mass <= 0.0)
        throw std::runtime_error("invalid integrator settings (steps/dt/mass) in " + path);
    if (m.nbins < 2 || m.s_hi <= m.s_lo)
        throw std::runtime_error("invalid bias grid (need nbins>=2, s_hi>s_lo) in " + path);
    if (m.deposit_every <= 0)
        throw std::runtime_error("invalid deposit pace (deposit_every<=0) in " + path);
    if (m.bias_factor <= 1.0)
        throw std::runtime_error("well-tempered MetaD needs bias_factor > 1 in " + path);
    if (c.n_walkers <= 0)
        throw std::runtime_error("invalid n_walkers (<=0) in " + path);

    return c;
}

// ---------------------------------------------------------------------------
// integrate_cpu: run the whole ensemble serially and form the mean bias grid.
//   Each walker is independent (its own bias grid, its own RNG stream), so this
//   is a plain loop -- which is exactly WHY it maps to one GPU thread per walker
//   in kernels.cu. We accumulate the per-bin SUM across walkers, then divide by
//   the walker count to get the ENSEMBLE-AVERAGE bias (the multi-walker estimate
//   of the free-energy surface, via metad::recover_fes()).
// ---------------------------------------------------------------------------
void integrate_cpu(const MetadConfig& c,
                   std::vector<metad::WalkerResult>& results,
                   std::vector<double>& mean_bias) {
    const int M = ensemble_size(c);
    const int nb = c.model.nbins;
    results.assign(static_cast<std::size_t>(M), metad::WalkerResult{});
    mean_bias.assign(static_cast<std::size_t>(nb), 0.0);

    // Scratch bias grid reused for each walker (run_walker zeroes it first).
    std::vector<double> bias(static_cast<std::size_t>(nb), 0.0);

    for (int id = 0; id < M; ++id) {
        // Integrate walker `id` fully; it fills `bias` with its accumulated hills.
        results[static_cast<std::size_t>(id)] =
            metad::run_walker(c.model, bias.data(), c.seed, id, walker_start(c, id));

        // Add this walker's bias grid into the running ensemble sum.
        for (int j = 0; j < nb; ++j)
            mean_bias[static_cast<std::size_t>(j)] += bias[static_cast<std::size_t>(j)];
    }

    // Divide by the number of walkers -> ensemble-average bias grid.
    const double inv_M = 1.0 / static_cast<double>(M);
    for (int j = 0; j < nb; ++j) mean_bias[static_cast<std::size_t>(j)] *= inv_M;
}
