// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial CPU reference (the trusted baseline)
// ---------------------------------------------------------------------------
// Project 1.25 : Gaussian-Accelerated MD (GaMD)   (reduced-scope teaching version)
//
// ROLE IN THE PROJECT
//   This is the "ground truth" the GPU result is checked against. It is written
//   to be OBVIOUSLY correct -- one walker after another in a plain loop, no
//   parallelism -- so that when the GPU and CPU tallies agree (here, EXACTLY,
//   because the accumulators are integers), we believe the GPU.
//
//   All the actual physics lives in gamd.h (shared host/device, PATTERNS §2);
//   this file only (a) parses the config and (b) drives run_walker() serially
//   with a plain integer-add functor. Compiled by the host C++ compiler only.
//
// READ THIS AFTER: reference_cpu.h, gamd.h. Compare against kernels.cu (GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error

// ---------------------------------------------------------------------------
// load_config: parse the 15-field whitespace-separated sample (data/README.md).
//   We read into named locals first, then validate, then return -- so a bad file
//   produces a precise error instead of a half-filled struct.
// ---------------------------------------------------------------------------
GamdConfig load_config(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open config file: " + path);

    GamdConfig c;
    // The read order MUST match the documented file format and data/README.md.
    if (!(in >> c.u_barrier >> c.kT >> c.gamma_fric >> c.dt >> c.steps >> c.equil_steps
             >> c.e_threshold >> c.v_min >> c.v_max >> c.k0
             >> c.n_walkers >> c.x_lo >> c.x_hi >> c.n_bins >> c.seed)) {
        throw std::runtime_error(
            "bad config (expected 15 fields: u_barrier kT gamma_fric dt steps "
            "equil_steps e_threshold v_min v_max k0 n_walkers x_lo x_hi n_bins seed) in " + path);
    }
    // Sanity checks: reject values that would break the integrator or histogram.
    if (c.kT <= 0.0 || c.gamma_fric <= 0.0 || c.dt <= 0.0 || c.steps <= 0
        || c.equil_steps < 0 || c.equil_steps >= c.steps
        || c.n_walkers <= 0 || c.n_bins <= 0 || c.x_hi <= c.x_lo
        || c.k0 <= 0.0 || c.k0 > 1.0 || c.v_max <= c.v_min) {
        throw std::runtime_error("invalid config values in " + path);
    }
    return c;
}

// ---------------------------------------------------------------------------
// run_ensemble_cpu: serial reference. One walker at a time, accumulating into
//   the SAME fixed-point integer array layout the kernel uses (count|S1|S2).
//   The lambda `add` is the CPU counterpart of the GPU's atomicAdd: since this
//   loop is single-threaded, a plain += is already correct AND deterministic.
// ---------------------------------------------------------------------------
void run_ensemble_cpu(const GamdConfig& c,
                      std::vector<int64_t>& acc,
                      std::vector<double>& final_x) {
    // Zero the accumulators: 3*n_bins int64 laid out [count | S1 | S2].
    acc.assign(static_cast<std::size_t>(acc_total(c)), 0);
    final_x.assign(static_cast<std::size_t>(c.n_walkers), 0.0);

    // Plain serial integer adder -- the host twin of the device atomicAdd.
    auto add = [&acc](int idx, int64_t v) {
        acc[static_cast<std::size_t>(idx)] += v;
    };

    // Drive each independent walker through the shared run_walker() loop. This is
    // the EXACT computation the kernel performs, one thread per walker.
    for (int w = 0; w < c.n_walkers; ++w) {
        final_x[static_cast<std::size_t>(w)] =
            run_walker(c, static_cast<uint32_t>(w), add);
    }
}
