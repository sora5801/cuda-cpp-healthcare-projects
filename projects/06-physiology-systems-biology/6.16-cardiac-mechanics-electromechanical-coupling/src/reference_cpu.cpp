// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial ensemble integration (the baseline)
// ---------------------------------------------------------------------------
// Project 6.16 : Cardiac Mechanics & Electromechanical Coupling
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- a single readable loop over the ensemble, no
//   parallelism -- so that when the GPU and CPU agree we believe the GPU. The
//   actual per-heart ODE/RK4 lives in cardiac.h and is shared verbatim with the
//   GPU kernel, so agreement is to round-off (PATTERNS.md section 2).
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: cardiac.h, reference_cpu.h. Compare with kernels.cu (GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// load_ensemble -- parse the whitespace-separated sample into an EnsembleConfig.
//   The file layout is documented in data/README.md. We read the baseline
//   physiology first, then the integration settings, then the sweep grid. If
//   anything is missing or non-physical we THROW so the demo fails loudly
//   instead of silently running on garbage.
// ---------------------------------------------------------------------------
EnsembleConfig load_ensemble(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open ensemble file: " + path);

    EnsembleConfig c;
    HeartParams& b = c.base;

    // -- Block 1: activation timing + calcium transient --------------------
    // t_activate  Ca_rest Ca_amp tau_rise tau_decay
    if (!(in >> b.t_activate
             >> b.Ca_rest >> b.Ca_amp >> b.tau_rise_ms >> b.tau_decay_ms))
        throw std::runtime_error("bad calcium block in " + path);

    // -- Block 2: cross-bridge / active state ------------------------------
    // Tref_baseline Ca50 nH k_xb    (Tref is overridden by the sweep)
    double Tref_baseline;   // read but overridden per member; kept for clarity
    if (!(in >> Tref_baseline >> b.Ca50 >> b.nH >> b.k_xb_ms))
        throw std::runtime_error("bad cross-bridge block in " + path);
    b.Tref = Tref_baseline;

    // -- Block 3: chamber elastance ----------------------------------------
    // Emin V0
    if (!(in >> b.Emin >> b.V0_mL))
        throw std::runtime_error("bad chamber-mechanics block in " + path);

    // -- Block 4: valves ----------------------------------------------------
    // R_ao R_mv P_ven
    if (!(in >> b.R_ao >> b.R_mv >> b.P_ven))
        throw std::runtime_error("bad valve block in " + path);

    // -- Block 5: Windkessel afterload -------------------------------------
    // R_sys_baseline C_art P_art_dias    (R_sys is overridden by the sweep)
    double Rsys_baseline;   // read but overridden per member
    if (!(in >> Rsys_baseline >> b.C_art >> b.P_art_dias))
        throw std::runtime_error("bad Windkessel block in " + path);
    b.R_sys = Rsys_baseline;

    // -- Block 6: integration settings -------------------------------------
    // dt steps_per_beat n_beats
    if (!(in >> c.dt_ms >> c.steps_per_beat >> c.n_beats))
        throw std::runtime_error("bad integration block in " + path);

    // -- Block 7: sweep grid -----------------------------------------------
    // nT nR  Tref_lo Tref_hi  R_lo R_hi     (contractility x afterload)
    if (!(in >> c.nT >> c.nR >> c.Tref_lo >> c.Tref_hi >> c.R_lo >> c.R_hi))
        throw std::runtime_error("bad sweep block in " + path);

    // -- Sanity checks so a typo can't silently produce nonsense -----------
    if (c.dt_ms <= 0.0 || c.steps_per_beat <= 0 || c.n_beats <= 0)
        throw std::runtime_error("invalid integration settings in " + path);
    if (c.nT <= 0 || c.nR <= 0)
        throw std::runtime_error("invalid sweep size in " + path);
    if (b.tau_rise_ms <= 0.0 || b.tau_decay_ms <= 0.0 || b.tau_rise_ms >= b.tau_decay_ms)
        throw std::runtime_error("need 0 < tau_rise < tau_decay in " + path);
    if (b.R_ao <= 0.0 || b.R_mv <= 0.0 || b.R_sys <= 0.0 || b.C_art <= 0.0)
        throw std::runtime_error("valve/Windkessel resistances and C must be positive in " + path);

    return c;
}

// ---------------------------------------------------------------------------
// integrate_cpu -- run every ensemble member serially.
//   Each member is an INDEPENDENT ODE solve -> a plain for-loop here, one GPU
//   thread per member in kernels.cu. This is the reference wall-time (timed in
//   main.cu) that makes the GPU speed-up legible.
// ---------------------------------------------------------------------------
void integrate_cpu(const EnsembleConfig& c, std::vector<CycleResult>& results) {
    const int M = ensemble_size(c);
    results.assign(static_cast<std::size_t>(M), CycleResult{});
    for (int idx = 0; idx < M; ++idx) {
        results[static_cast<std::size_t>(idx)] = integrate_member(c, idx);
    }
}
