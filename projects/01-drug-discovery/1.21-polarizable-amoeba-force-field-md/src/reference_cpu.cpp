// ===========================================================================
// src/reference_cpu.cpp  --  Loader, synthetic builder, serial CG reference
// ---------------------------------------------------------------------------
// Project 1.21 : Polarizable / AMOEBA Force Field MD
//
// ROLE IN THE PROJECT
//   The "ground truth" the GPU result is checked against. It is deliberately
//   plain -- a single loop over ensemble members calling the SAME CG solver the
//   GPU kernel uses (solve_induced_dipoles in amoeba.h) -- so that when GPU and
//   CPU agree, we trust the GPU. Compiled by the host C++ compiler only (no CUDA
//   here); the shared physics is host+device via AMOEBA_HD.
//
// READ THIS AFTER: reference_cpu.h, amoeba.h. Compare against kernels.cu.
// ===========================================================================
#include "reference_cpu.h"

#include <fstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// load_ensemble: parse the data/sample text format into an EnsembleConfig.
//   We read field by field with operator>> (whitespace-insensitive) and validate
//   every count so a truncated file cannot silently produce a half-built system.
// ---------------------------------------------------------------------------
EnsembleConfig load_ensemble(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open ensemble file: " + path);

    EnsembleConfig c;
    int M = 0;
    if (!(in >> M >> c.tol >> c.max_iter) || M <= 0)
        throw std::runtime_error("bad header (expected 'M tol max_iter') in " + path);

    c.systems.resize(static_cast<std::size_t>(M));
    for (int m = 0; m < M; ++m) {
        AtomSystem& s = c.systems[static_cast<std::size_t>(m)];
        if (!(in >> s.n) || s.n <= 0 || s.n > AMOEBA_MAX_ATOMS)
            throw std::runtime_error("bad atom count for a system in " + path);
        for (int i = 0; i < s.n; ++i) {
            // Per atom: position (x y z), permanent field (Ex Ey Ez), polarizability.
            if (!(in >> s.pos[i][0] >> s.pos[i][1] >> s.pos[i][2]
                     >> s.Eperm[i][0] >> s.Eperm[i][1] >> s.Eperm[i][2]
                     >> s.alpha[i]))
                throw std::runtime_error("truncated atom record in " + path);
        }
    }
    return c;
}

// ---------------------------------------------------------------------------
// make_synthetic_ensemble: deterministic, clearly-SYNTHETIC ensemble.
//   Geometry per member: a "water trio" -- one central polarizable site at the
//   origin and two partner sites placed symmetrically along +/- x at a distance
//   `d` that DECREASES across the ensemble. As the partners approach, the
//   dipole-dipole coupling (which scales like 1/d^3) strengthens, so the induced
//   dipoles and the polarization energy grow -- the exact qualitative behaviour
//   the learner should expect from a polarizable model. A fixed permanent field
//   points along +x so the result is a clean, signed dipole the demo can report.
//
//   Everything is computed from integer member indices with no RNG -> the same
//   ensemble every run (determinism, PATTERNS.md section 3).
// ---------------------------------------------------------------------------
EnsembleConfig make_synthetic_ensemble(int members) {
    EnsembleConfig c;
    c.tol      = 1.0e-8;
    c.max_iter = 64;
    if (members < 1) members = 1;
    c.systems.resize(static_cast<std::size_t>(members));

    for (int m = 0; m < members; ++m) {
        AtomSystem& s = c.systems[static_cast<std::size_t>(m)];
        s.n = 3;

        // Separation sweeps from 4.0 Angstrom (weak coupling) down to ~2.0
        // (strong coupling) as m runs 0..members-1. The closed-form keeps it
        // deterministic and easy to reason about.
        const double dmax = 4.0, dmin = 2.0;
        const double frac = (members > 1) ? double(m) / double(members - 1) : 0.0;
        const double d    = dmax - (dmax - dmin) * frac;   // [Angstrom]

        // Central atom at origin; partners at (+/- d, 0, 0).
        s.pos[0][0] = 0.0;  s.pos[0][1] = 0.0;  s.pos[0][2] = 0.0;
        s.pos[1][0] =  d;   s.pos[1][1] = 0.0;  s.pos[1][2] = 0.0;
        s.pos[2][0] = -d;   s.pos[2][1] = 0.0;  s.pos[2][2] = 0.0;

        // A uniform external permanent field along +x (magnitude 0.05 field
        // units) acts on every atom. (In a full model this would come from the
        // permanent multipoles of neighbours; here it is a clean fixed driver.)
        for (int i = 0; i < s.n; ++i) {
            s.Eperm[i][0] = 0.05;  s.Eperm[i][1] = 0.0;  s.Eperm[i][2] = 0.0;
            // Polarizability ~ water oxygen (~1.4 Angstrom^3); the partners are a
            // touch smaller so the three dipoles are distinguishable.
            s.alpha[i] = (i == 0) ? 1.40 : 1.10;
        }
    }
    return c;
}

// ---------------------------------------------------------------------------
// integrate_cpu: serial reference. One member at a time, same CG as the GPU.
//   This loop is the "before" picture for the GPU: members are independent, so
//   the only thing the GPU changes is running them all at once (kernels.cu).
// ---------------------------------------------------------------------------
void integrate_cpu(const EnsembleConfig& c, std::vector<PerSystemResult>& results) {
    const int M = ensemble_size(c);
    results.assign(static_cast<std::size_t>(M), PerSystemResult{});
    double mu[AMOEBA_MAX_ATOMS][3];   // scratch dipoles, reused per member
    for (int m = 0; m < M; ++m) {
        results[static_cast<std::size_t>(m)] =
            solve_induced_dipoles(c.systems[static_cast<std::size_t>(m)],
                                  c.tol, c.max_iter, mu);
    }
}
