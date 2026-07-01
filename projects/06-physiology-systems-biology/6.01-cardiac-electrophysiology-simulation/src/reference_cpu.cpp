// ===========================================================================
// src/reference_cpu.cpp  --  Loader + serial monodomain reference solver
// ---------------------------------------------------------------------------
// Project 6.1 : Cardiac Electrophysiology Simulation
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU result is checked against. It is written to be
//   OBVIOUSLY correct -- plain nested loops, no parallelism -- so that when the
//   GPU and CPU agree, we trust the GPU. Compiled by the host C++ compiler only
//   (no CUDA here). The per-cell physics lives in the shared cardiac_cell.h, so
//   this file just orchestrates the operator-split time loop over the grid.
//
// READ THIS AFTER: reference_cpu.h. Compare against kernels.cu (the GPU twin).
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// load_monodomain: parse the one-line-ish sample file. The field order is fixed
//   (see reference_cpu.h / data/README.md). We validate the physically-required
//   invariants (positive grid, dt within a sane range, D>0) and throw on any
//   violation so a bad sample cannot silently produce nonsense.
// ---------------------------------------------------------------------------
MonodomainParams load_monodomain(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open monodomain parameter file: " + path);

    MonodomainParams p;
    // Stream the 14 fields in order. operator>> skips whitespace/newlines, so
    // the file may be one line or spread across several -- both parse the same.
    if (!(in >> p.nx >> p.ny >> p.steps
             >> p.dt >> p.dx >> p.D
             >> p.a  >> p.eps >> p.b
             >> p.stim_x0 >> p.stim_y0 >> p.stim_w >> p.stim_h >> p.stim_v)) {
        throw std::runtime_error(
            "bad parameters (expected 14 fields: nx ny steps dt dx D a eps b "
            "stim_x0 stim_y0 stim_w stim_h stim_v) in " + path);
    }

    // --- Validate: a mistake here would crash the kernel or diverge silently.
    if (p.nx <= 0 || p.ny <= 0 || p.steps < 0)
        throw std::runtime_error("invalid grid/steps (need nx>0, ny>0, steps>=0) in " + path);
    if (p.dt <= 0.0 || p.dx <= 0.0 || p.D <= 0.0)
        throw std::runtime_error("invalid numerics (need dt>0, dx>0, D>0) in " + path);
    if (p.a <= 0.0 || p.a >= 1.0)
        throw std::runtime_error("invalid FHN threshold a (need 0<a<1) in " + path);
    // Warn-by-throw if the explicit diffusion step is unstable (CFL). We refuse
    // to run an unstable sample rather than emit exploding, meaningless numbers.
    if (p.dt > cfl_limit(p))
        throw std::runtime_error("dt exceeds the explicit-diffusion CFL limit "
                                 "dx^2/(4D); reduce dt or D in " + path);
    return p;
}

// ---------------------------------------------------------------------------
// init_state: the shared initial condition. Everything at rest (V=0, w=0),
//   except a small square S1 patch of tissue that is depolarised to stim_v.
//   That patch is the "pacemaker" spark that launches a travelling wave.
//   Clamped to the grid so an out-of-range stimulus box cannot write OOB.
// ---------------------------------------------------------------------------
void init_state(const MonodomainParams& p,
                std::vector<double>& V, std::vector<double>& w) {
    const std::size_t cells = static_cast<std::size_t>(p.nx) * p.ny;
    V.assign(cells, 0.0);   // resting transmembrane voltage
    w.assign(cells, 0.0);   // resting recovery variable

    for (int y = p.stim_y0; y < p.stim_y0 + p.stim_h; ++y) {
        if (y < 0 || y >= p.ny) continue;
        for (int x = p.stim_x0; x < p.stim_x0 + p.stim_w; ++x) {
            if (x < 0 || x >= p.nx) continue;
            V[cell_idx(x, y, p.nx)] = p.stim_v;   // spark the excitation
        }
    }
}

// ---------------------------------------------------------------------------
// monodomain_cpu: the serial reference solver. Operator splitting (Godunov):
//   for each timestep:
//     (A) REACTION half-step -- update every cell's (V,w) IN PLACE via the local
//         FHN ODE (react_step). Purely pointwise; order does not matter.
//     (B) DIFFUSION half-step -- compute the 5-point Laplacian update for every
//         cell, reading the post-reaction V from a read-only buffer and writing
//         to a second buffer, then SWAP (ping-pong). The read/write separation
//         is exactly what makes the GPU version race-free.
//
//   Complexity: O(steps * nx * ny) time, O(nx*ny) space. This nested loop is the
//   serial baseline whose wall time (timed in main.cu) contrasts with the GPU.
// ---------------------------------------------------------------------------
void monodomain_cpu(const MonodomainParams& p,
                    std::vector<double>& V_final, std::vector<double>& w_final) {
    const std::size_t cells = static_cast<std::size_t>(p.nx) * p.ny;

    std::vector<double> V, w;
    init_state(p, V, w);
    std::vector<double> V_next(cells, 0.0);   // scratch buffer for diffusion

    for (int s = 0; s < p.steps; ++s) {
        // (A) REACTION: pointwise ODE, updates V and w in place.
        for (int y = 0; y < p.ny; ++y)
            for (int x = 0; x < p.nx; ++x)
                react_step(&V[cell_idx(x, y, p.nx)],
                           &w[cell_idx(x, y, p.nx)], p);

        // (B) DIFFUSION: read V, write V_next (ping-pong), then swap the buffers.
        for (int y = 0; y < p.ny; ++y)
            for (int x = 0; x < p.nx; ++x)
                V_next[cell_idx(x, y, p.nx)] = diffuse_cell(x, y, V.data(), p);
        V.swap(V_next);   // V now holds the post-diffusion field
    }

    V_final = V;
    w_final = w;
}
