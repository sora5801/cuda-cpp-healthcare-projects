// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial CPU reference for the DFT sweep
// ---------------------------------------------------------------------------
// Project 6.19 : Defibrillation & High-Voltage Shock Simulation
//
// Compiled by the HOST C++ compiler only (no CUDA here). Every bit of per-step
// physics is delegated to the shared defib.h functions, so this reference and
// the GPU kernel produce identical numbers -- the whole point of the shared
// host/device core (docs/PATTERNS.md section 2). It is written to be OBVIOUSLY
// correct: readable loops, no parallelism, no cleverness -- so when the GPU
// agrees with it, we believe the GPU.
//
// Read defib.h and reference_cpu.h first; this file just wires them together
// into a serial baseline plus small loaders/reducers.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// load_sweep: read the three-line sample format (see data/README.md). We read
// field by field and validate aggressively so a corrupt sample fails with a
// clear message rather than silently simulating garbage.
// ---------------------------------------------------------------------------
ShockSweep load_sweep(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open sample file: " + path);

    ShockSweep s;
    FhnParams& p = s.p;

    // Line 1: the cable + FHN parameters.
    if (!(in >> p.ncell >> p.nsteps >> p.dt >> p.dx >> p.D >> p.a >> p.eps >> p.gamma))
        throw std::runtime_error("bad line 1 (need 'ncell nsteps dt dx D a eps gamma') in " + path);

    // Line 2: the initial condition + shock protocol + success threshold.
    if (!(in >> p.initial_excited >> p.shock_start >> p.shock_len
             >> p.biphasic >> s.success_thresh))
        throw std::runtime_error("bad line 2 (need 'init_excited shock_start shock_len biphasic thresh') in " + path);

    // Line 3: the number of amplitudes, then that many amplitude values.
    int namp = 0;
    if (!(in >> namp) || namp <= 0)
        throw std::runtime_error("bad amplitude count on line 3 in " + path);
    s.amps.resize(static_cast<std::size_t>(namp));
    for (int k = 0; k < namp; ++k) {
        if (!(in >> s.amps[static_cast<std::size_t>(k)]))
            throw std::runtime_error("missing amplitude value in " + path);
    }

    // --- Physical sanity + numerical stability checks -----------------------
    if (p.ncell <= 1 || p.nsteps <= 0 || p.dt <= 0.0 || p.dx <= 0.0 || p.D <= 0.0)
        throw std::runtime_error("invalid grid/time parameters in " + path);
    // Explicit diffusion is stable only if dt <= dx^2 / (2 D) (the CFL-like
    // limit for the forward-Euler heat equation -- see THEORY.md numerics).
    const double dt_max = (p.dx * p.dx) / (2.0 * p.D);
    if (p.dt > dt_max)
        throw std::runtime_error("dt exceeds diffusion stability limit dx^2/(2D) in " + path);
    if (p.initial_excited < 0 || p.initial_excited > p.ncell)
        throw std::runtime_error("initial_excited out of range in " + path);
    return s;
}

// ---------------------------------------------------------------------------
// simulate_one_cpu: one full cable simulation for a single shock amplitude.
//   State layout: two flat arrays V[ncell], w[ncell], double-buffered so each
//   step reads the OLD field and writes the NEW field (ping-pong), matching the
//   GPU exactly. Returns the residual activity after the last step.
//   Complexity: O(nsteps * ncell) time, O(ncell) space.
// ---------------------------------------------------------------------------
double simulate_one_cpu(const FhnParams& p, double amp) {
    const int n = p.ncell;

    // Two buffers for V and two for w; we swap pointers each step. Starting with
    // std::vector gives us zero-initialised, correctly-sized storage.
    std::vector<double> Va(n, 0.0), Vb(n, 0.0), wa(n, 0.0), wb(n, 0.0);

    // Initial condition: the left `initial_excited` cells start fully excited
    // (V=1), the rest at rest (V=0). This seeds a propagating wavefront -- the
    // ongoing electrical activity that the shock must terminate.
    for (int i = 0; i < p.initial_excited; ++i) Va[i] = 1.0;

    double* Vsrc = Va.data();  double* Vdst = Vb.data();
    double* wsrc = wa.data();  double* wdst = wb.data();

    // Time loop: advance the whole cable one step at a time (ping-pong buffers).
    for (int s = 0; s < p.nsteps; ++s) {
        cable_step(s, amp, p, Vsrc, wsrc, Vdst, wdst);   // shared physics
        // Swap so the freshly-written field becomes the input to the next step.
        double* tV = Vsrc; Vsrc = Vdst; Vdst = tV;
        double* tw = wsrc; wsrc = wdst; wdst = tw;
    }

    // `Vsrc` now points at the latest voltage field (after the final swap).
    return activity_metric(n, Vsrc);
}

// ---------------------------------------------------------------------------
// sweep_cpu: run one simulation per amplitude. Embarrassingly parallel across
// amplitudes -- but here it is a simple serial loop (the trusted baseline).
// ---------------------------------------------------------------------------
void sweep_cpu(const ShockSweep& s, std::vector<double>& residual) {
    residual.assign(s.amps.size(), 0.0);
    for (std::size_t k = 0; k < s.amps.size(); ++k)
        residual[k] = simulate_one_cpu(s.p, s.amps[k]);
}

// ---------------------------------------------------------------------------
// find_dft: the smallest-amplitude successful shock. Amplitudes are ascending,
// so we scan from the weakest and return the first index whose residual falls
// below the success threshold. -1 means no tested shock defibrillated.
// ---------------------------------------------------------------------------
int find_dft(const ShockSweep& s, const std::vector<double>& residual) {
    for (std::size_t k = 0; k < residual.size(); ++k)
        if (residual[k] < s.success_thresh)
            return static_cast<int>(k);
    return -1;
}
