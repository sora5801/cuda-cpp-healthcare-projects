// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial per-sinusoid perfusion solve
// ---------------------------------------------------------------------------
// Project 6.25 : Liver & Kidney Perfusion Modeling
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU result is checked against. It is written to be
//   OBVIOUSLY correct -- a single readable loop over sinusoids, no parallelism --
//   so that when the GPU and CPU agree we believe the GPU. The actual transport
//   ODE/RK4 lives in perfusion.h and is SHARED with the kernel, so agreement is
//   to round-off (PATTERNS.md section 2).
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, perfusion.h. Compare with kernels.cu (twin).
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// load_lobule: read the whitespace text format (data/README.md):
//   "L C_in Km Vmax_pp Vmax_cl nseg   nsin v_lo v_hi"
//   Lines beginning with '#' are comments and are skipped, so the committed
//   sample can carry a self-documenting header. Fails LOUDLY (throws) on a
//   missing file or physically invalid values so a demo never silently runs on
//   garbage.
// ---------------------------------------------------------------------------
LobuleConfig load_lobule(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open lobule file: " + path);

    // Concatenate all NON-comment lines into one stream, then parse the fields.
    // This keeps the numeric parse identical to a single-line file while letting
    // the sample document its own layout with leading '#' lines.
    std::ostringstream body;
    std::string line;
    while (std::getline(in, line)) {
        std::size_t first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) continue;      // blank line
        if (line[first] == '#') continue;              // comment line
        body << line << ' ';
    }

    std::istringstream fields(body.str());
    LobuleConfig c;
    if (!(fields >> c.p.L >> c.p.C_in >> c.p.Km >> c.p.Vmax_pp >> c.p.Vmax_cl >> c.p.nseg
                 >> c.nsin >> c.v_lo >> c.v_hi))
        throw std::runtime_error("bad parameters (expected "
            "'L C_in Km Vmax_pp Vmax_cl nseg nsin v_lo v_hi') in " + path);

    // Physical sanity: every length/rate must be positive, the grid non-empty,
    // and the velocity sweep valid (v_lo>0 so dividing by v in dCdx is safe).
    if (c.p.L <= 0.0 || c.p.C_in < 0.0 || c.p.Km <= 0.0 || c.p.nseg <= 0
        || c.nsin <= 0 || c.v_lo <= 0.0 || c.v_hi <= 0.0)
        throw std::runtime_error("invalid (non-positive) lobule parameters in " + path);
    return c;
}

// ---------------------------------------------------------------------------
// integrate_cpu: solve every sinusoid serially. Each member is an INDEPENDENT
//   1-D transport-reaction ODE (one plain loop here; one GPU thread per sinusoid
//   in kernels.cu). The per-sinusoid velocity comes from the sweep mapping in
//   reference_cpu.h; the physics/RK4 comes from perfusion.h -- so this loop and
//   the kernel compute byte-identical numbers.
// ---------------------------------------------------------------------------
void integrate_cpu(const LobuleConfig& c, std::vector<SinusoidResult>& results) {
    const int M = lobule_size(c);
    results.assign(static_cast<std::size_t>(M), SinusoidResult{});
    for (int idx = 0; idx < M; ++idx) {
        const double v = sinusoid_velocity(c, idx);          // this sinusoid's blood velocity
        results[static_cast<std::size_t>(idx)] = integrate_sinusoid(c.p, v);
    }
}
