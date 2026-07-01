// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial track-structure reference
// ---------------------------------------------------------------------------
// Project 5.11 : Microdosimetry & Track-Structure Simulation
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- a single readable loop over tracks, no parallelism
//   -- so that when the GPU and CPU agree, we believe the GPU.
//
//   Compiled by the host C++ compiler only (no CUDA here). The actual physics
//   lives in ts_physics.h, shared with the GPU kernel, so the two run identical
//   histories. See reference_cpu.h and compare against kernels.cu (the GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// load_track_problem: read the one-line whitespace-separated parameter file.
//   The field order is fixed and documented in data/README.md. We validate the
//   physically meaningful constraints (positive sizes, at least one track) so the
//   demo fails loudly on a malformed file instead of silently producing garbage.
// ---------------------------------------------------------------------------
TrackProblem load_track_problem(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open parameter file: " + path);

    TrackProblem p;
    if (!(in >> p.tp.box_nm >> p.tp.sigma_ion >> p.tp.let_spread >> p.tp.quantum_eV
             >> p.tp.quanta_per_ion >> p.tp.p_delta >> p.tp.delta_quanta
             >> p.tp.dna_radius_nm >> p.tp.n_dna_segments >> p.tp.n_y_bins
             >> p.tp.y_max_keV_um >> p.n_tracks >> p.seed))
        throw std::runtime_error(
            "bad parameters (expected 'box_nm sigma_ion let_spread quantum_eV "
            "quanta_per_ion p_delta delta_quanta dna_radius_nm n_dna_segments "
            "n_y_bins y_max_keV_um n_tracks seed') in " + path);

    if (p.tp.box_nm <= 0 || p.tp.sigma_ion <= 0 || p.tp.n_dna_segments <= 0 ||
        p.tp.n_y_bins <= 0 || p.tp.y_max_keV_um <= 0 || p.n_tracks == 0)
        throw std::runtime_error("invalid simulation parameters in " + path);

    return p;
}

// ---------------------------------------------------------------------------
// track_cpu: simulate every primary track serially and accumulate the tallies.
//   Each track gets its own reproducible RNG stream seeded from (seed, i), runs
//   the shared transport, and its integer outputs are added to the running sums.
//   This is EXACTLY what the GPU does -- just serially and with plain '+=' in
//   place of atomicAdd. Integer counts make the two sums bit-identical.
//   Complexity: O(n_tracks * steps_per_track), fully serial -- the baseline whose
//   wall time (timed in main.cu) makes the GPU speed-up legible.
// ---------------------------------------------------------------------------
void track_cpu(const TrackProblem& prob, TrackTally& tally) {
    tally.total_quanta = 0;
    tally.total_ssb    = 0;
    tally.total_dsb    = 0;
    tally.y_hist.assign(static_cast<std::size_t>(prob.tp.n_y_bins), 0ULL);

    for (unsigned long long i = 0; i < prob.n_tracks; ++i) {
        Rng rng = rng_seed(prob.seed, i);            // this track's private stream
        TrackResult r = simulate_track(prob.tp, rng); // shared physics

        tally.total_quanta += r.energy_quanta;
        tally.total_ssb    += static_cast<unsigned long long>(r.ssb);
        tally.total_dsb    += static_cast<unsigned long long>(r.dsb);
        tally.y_hist[static_cast<std::size_t>(r.y_bin)] += 1ULL;
    }
}
