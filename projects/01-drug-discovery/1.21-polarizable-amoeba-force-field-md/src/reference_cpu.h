// ===========================================================================
// src/reference_cpu.h  --  Ensemble config + CPU reference for AMOEBA dipoles
// ---------------------------------------------------------------------------
// Project 1.21 : Polarizable / AMOEBA Force Field MD
//
// In a real AMOEBA MD run the induced-dipole CG solve is re-run at EVERY time
// step (the atoms moved, so the permanent field and the coupling tensor changed).
// We mirror that workload as an ENSEMBLE of independent polarization systems:
// each "member" is one molecular configuration (positions + permanent field +
// polarizabilities) for which we solve A mu = b. Solving one member is sequential
// (the CG loop), but members are independent of each other -> one GPU thread per
// member (PATTERNS.md "ensemble" pattern, exemplified by flagship 9.02).
//
// The config + the synthetic-builder + the loader live here; the actual physics
// and the CG solver are in amoeba.h. Pure C++ -- kernels.cu reuses these types.
//
// READ THIS AFTER: amoeba.h. Compare reference_cpu.cpp against kernels.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "amoeba.h"   // AMOEBA_HD, AtomSystem, PerSystemResult, solve_induced_dipoles

// ---------------------------------------------------------------------------
// EnsembleConfig  --  the whole batch of polarization systems plus solver knobs.
//   `systems` is a flat array of AtomSystem; member index = position in it.
//   Storing them contiguously lets us cudaMemcpy the entire ensemble to the GPU
//   in a single transfer (see kernels.cu).
// ---------------------------------------------------------------------------
struct EnsembleConfig {
    std::vector<AtomSystem> systems;   // one polarization problem per member
    double tol      = 1.0e-8;          // CG relative-residual stop (see amoeba.h)
    int    max_iter = 64;              // CG iteration cap (<= 3*AMOEBA_MAX_ATOMS)
};

// Number of ensemble members (independent CG solves).
inline int ensemble_size(const EnsembleConfig& c) {
    return static_cast<int>(c.systems.size());
}

// ---------------------------------------------------------------------------
// load_ensemble: read an ensemble from the tiny text format in data/sample.
//   Format (whitespace-separated, documented in data/README.md):
//       M tol max_iter
//       <repeated M times:>
//          n                                   (atoms in this system)
//          <repeated n times:>  x y z  Ex Ey Ez  alpha
//   Throws std::runtime_error on a missing/malformed file so the demo fails
//   loudly rather than silently running on garbage.
// ---------------------------------------------------------------------------
EnsembleConfig load_ensemble(const std::string& path);

// ---------------------------------------------------------------------------
// make_synthetic_ensemble: build a deterministic, clearly-SYNTHETIC ensemble in
//   memory (used when no data file is supplied). Each member is a small water-
//   like cluster; one physical knob -- the separation between two polarizable
//   atoms -- is swept across members so the demo shows the induced dipoles (and
//   the polarization energy) growing as atoms approach. See data/README.md.
// ---------------------------------------------------------------------------
EnsembleConfig make_synthetic_ensemble(int members);

// ---------------------------------------------------------------------------
// integrate_cpu: the trusted serial baseline. Solve every member's induced
//   dipoles with the SAME CG routine the GPU kernel calls (amoeba.h), filling
//   `results` (sized to M). main.cu compares this against the GPU output.
// ---------------------------------------------------------------------------
void integrate_cpu(const EnsembleConfig& c, std::vector<PerSystemResult>& results);
