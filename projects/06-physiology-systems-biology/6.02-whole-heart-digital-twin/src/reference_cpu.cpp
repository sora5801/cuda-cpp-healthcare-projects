// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial ensemble of whole-heart twins
// ---------------------------------------------------------------------------
// Project 6.2 : Whole-Heart Digital Twin   (REDUCED-SCOPE TEACHING VERSION)
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- a single readable loop over ensemble members, no
//   parallelism -- so that when the GPU and CPU agree we trust the GPU. The
//   actual heart physics (FHN EP + elastance mechanics + Windkessel) is in
//   heart.h and is SHARED with the kernel, so the two match to round-off.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, heart.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error

// ---------------------------------------------------------------------------
// load_ensemble -- parse the tiny whitespace-separated config sample.
//   Layout (see data/README.md):
//     n emax_lo emax_hi dt_ms beats target_sv bcl_ms E_min V0 Rp C_art
//   The last five fields override the corresponding HeartParams defaults so a
//   learner can perturb the physiology from the data file without recompiling.
// ---------------------------------------------------------------------------
EnsembleConfig load_ensemble(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open ensemble file: " + path);

    EnsembleConfig c;
    // These locals mirror the on-disk order; we read into them then validate.
    double bcl_ms = 0.0, e_min = 0.0, v0 = 0.0, rp = 0.0, c_art = 0.0;
    if (!(in >> c.n >> c.emax_lo >> c.emax_hi >> c.dt_ms >> c.beats
             >> c.target_sv >> bcl_ms >> e_min >> v0 >> rp >> c_art)) {
        throw std::runtime_error(
            "bad parameters (expected 'n emax_lo emax_hi dt_ms beats target_sv "
            "bcl_ms E_min V0 Rp C_art') in " + path);
    }

    // Fold the file-provided physiology into the baseline HeartParams.
    c.base.bcl_ms = bcl_ms;
    c.base.E_min  = e_min;
    c.base.V0     = v0;
    c.base.Rp     = rp;
    c.base.C_art  = c_art;

    // Basic sanity so the demo fails loudly on nonsense rather than NaNs later.
    if (c.n <= 0 || c.dt_ms <= 0.0 || c.beats <= 0 || c.emax_hi < c.emax_lo)
        throw std::runtime_error("invalid ensemble parameters in " + path);

    return c;
}

// ---------------------------------------------------------------------------
// integrate_cpu -- simulate every ensemble member serially.
//   Each member is an INDEPENDENT forward heart solve, so this is a plain loop
//   here and one GPU thread per member in kernels.cu. Complexity: O(n * steps)
//   where steps = beats * (bcl_ms/dt_ms); each member costs the same, so the
//   GPU's win grows linearly with the ensemble size n.
// ---------------------------------------------------------------------------
void integrate_cpu(const EnsembleConfig& c, std::vector<TwinResult>& results) {
    const int n = ensemble_size(c);
    results.assign(static_cast<std::size_t>(n), TwinResult{});
    for (int idx = 0; idx < n; ++idx) {
        // Build this member's heart (baseline physiology + its own E_max) ...
        const HeartParams p = member_params(c, idx);
        // ... and run the shared forward model to a steady-state summary.
        results[static_cast<std::size_t>(idx)] = simulate_heart(p, c.dt_ms, c.beats);
    }
}
