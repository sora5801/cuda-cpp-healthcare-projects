// ===========================================================================
// src/reference_cpu.cpp  --  Loader, field init, and the serial CPU reference
// ---------------------------------------------------------------------------
// Project 6.8 : Tumor Growth & Treatment-Response Modeling
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU result is checked against. It is written to be
//   OBVIOUSLY correct -- readable nested loops, no parallelism -- so that when
//   the GPU and CPU agree, we believe the GPU. The per-cell physics is NOT
//   duplicated here: it is the shared tumor.h, called from a plain loop, exactly
//   as the kernel calls it from one thread. That is what makes the comparison in
//   main.cu an exact test rather than a rough one.
//
//   Compiled by the host C++ compiler only (no CUDA here). See reference_cpu.h.
//
// READ THIS AFTER: reference_cpu.h, tumor.h. Compare against kernels.cu (twin).
// ===========================================================================
#include "reference_cpu.h"

#include <cmath>       // std::sqrt for the seed disc, std::exp via tumor.h
#include <fstream>     // std::ifstream
#include <stdexcept>   // std::runtime_error
#include <utility>     // std::swap

// ---------------------------------------------------------------------------
// load_tumor: parse the one-line parameter file into a TumorParams.
//   Format (whitespace-separated, see data/README.md):
//     nx ny dx D rho dt steps alpha beta dose n_fractions fx_interval seed_radius seed_u
//   We validate aggressively: a PDE run with a bad timestep produces garbage
//   (numerical blow-up), so we'd rather fail loudly at load time.
// ---------------------------------------------------------------------------
TumorParams load_tumor(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open tumor parameter file: " + path);

    TumorParams P{};
    if (!(in >> P.nx >> P.ny >> P.dx >> P.D >> P.rho >> P.dt >> P.steps
             >> P.alpha >> P.beta >> P.dose >> P.n_fractions >> P.fx_interval
             >> P.seed_radius >> P.seed_u)) {
        throw std::runtime_error(
            "bad parameters (expected 'nx ny dx D rho dt steps alpha beta dose "
            "n_fractions fx_interval seed_radius seed_u') in " + path);
    }

    // Basic sanity on shapes and positivity.
    if (P.nx <= 2 || P.ny <= 2 || P.steps < 0 || P.dx <= 0.0 || P.dt <= 0.0)
        throw std::runtime_error("invalid tumor grid/time parameters in " + path);
    if (P.D < 0.0 || P.rho < 0.0 || P.seed_u < 0.0 || P.seed_u > 1.0)
        throw std::runtime_error("invalid tumor physics parameters in " + path);

    // Explicit-Euler stability for 2-D diffusion: dt must satisfy
    //   dt <= dx^2 / (4 D)   (the diffusion-number / CFL condition).
    // Violating it makes the field oscillate and explode; catch it here.
    if (P.D > 0.0) {
        const double dt_max = (P.dx * P.dx) / (4.0 * P.D);
        if (P.dt > dt_max) {
            throw std::runtime_error(
                "unstable timestep: dt exceeds dx^2/(4D) stability limit in " + path);
        }
    }
    return P;
}

// ---------------------------------------------------------------------------
// init_field: build the initial tumor. u = 0 everywhere except a filled disc of
//   radius seed_radius [mm] at the grid centre, set to seed_u. Using a disc (not
//   a single point) gives a smooth front that the Fisher-KPP wave sharpens into
//   a travelling front -- the classic behaviour we want to show.
// ---------------------------------------------------------------------------
void init_field(const TumorParams& P, std::vector<double>& u) {
    const int N = P.nx * P.ny;
    u.assign(static_cast<std::size_t>(N), 0.0);          // healthy tissue: no tumor

    const double cx = 0.5 * (P.nx - 1);                  // grid-centre (cell units)
    const double cy = 0.5 * (P.ny - 1);
    const double r_cells = P.seed_radius / P.dx;         // seed radius in cell units
    const double r2 = r_cells * r_cells;

    for (int y = 0; y < P.ny; ++y) {
        for (int x = 0; x < P.nx; ++x) {
            const double ddx = x - cx, ddy = y - cy;      // offset from centre
            if (ddx * ddx + ddy * ddy <= r2) {            // inside the seed disc?
                u[static_cast<std::size_t>(y) * P.nx + x] = P.seed_u;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// simulate_cpu: the serial reference time loop.
//   For each step s:
//     (a) if s is a scheduled fraction, apply the LQ treatment kill everywhere;
//     (b) advance every cell one Fisher-KPP growth step (double-buffered).
//   Treatment is applied IN PLACE on the current buffer (it is a pure per-cell
//   multiply, no neighbours), then growth reads that buffer and writes the other.
//   Ping-ponging (std::swap of the two pointers) means "next state" is computed
//   from the frozen "current state", exactly like the GPU kernel -- so there is
//   no read/write hazard and the two paths match.
// ---------------------------------------------------------------------------
void simulate_cpu(const TumorParams& P, std::vector<double>& u) {
    const int N = P.nx * P.ny;
    std::vector<double> ub(static_cast<std::size_t>(N));  // scratch "next" buffer
    double* us = u.data();                                // source (current state)
    double* ud = ub.data();                               // destination (next state)

    for (int s = 0; s < P.steps; ++s) {
        // (a) Treatment: multiply every cell by the LQ surviving fraction. We
        //     compute S once (identical for all cells) then apply it in place.
        if (is_fraction_step(P, s)) {
            const double S = lq_survival(P.alpha, P.beta, P.dose);
            for (int i = 0; i < N; ++i) tumor_treat_update(i, S, us);
        }
        // (b) Growth: one explicit-Euler Fisher-KPP step for every cell.
        for (int y = 0; y < P.ny; ++y)
            for (int x = 0; x < P.nx; ++x)
                tumor_grow_update(x, y, P, us, ud);
        std::swap(us, ud);   // the freshly written buffer becomes "current"
    }

    // After the final swap, `us` points at the latest state. If that is the
    // scratch buffer (odd step count), copy it back so the caller sees the result
    // through the vector it passed in.
    if (us != u.data()) u.assign(us, us + N);
}
