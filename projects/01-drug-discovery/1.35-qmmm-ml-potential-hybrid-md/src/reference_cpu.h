// ===========================================================================
// src/reference_cpu.h  --  Ensemble config + CPU reference (the trusted baseline)
// ---------------------------------------------------------------------------
// Project 1.35 : QMMM/ML Potential Hybrid MD   (reduced-scope teaching version)
//
// The ensemble is a set of M INDEPENDENT short MD trajectories that differ only
// by a fixed per-member perturbation of the link atom (an active-learning probe
// of configuration space). The trajectory driver itself lives in nnpmm.h (shared
// host+device). This header adds the host-only glue:
//   * EnsembleConfig          -- how many members + integration settings,
//   * load_ensemble()         -- read it from the tiny text sample,
//   * integrate_cpu()         -- the serial reference the GPU is checked against.
//
// Pure C++ (no CUDA): kernels.cu reuses EnsembleConfig and TrajResult unchanged.
// READ THIS AFTER: nnpmm.h. Compare against kernels.cu (the GPU twin).
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "nnpmm.h"   // NNPMM_HD, run_trajectory, TrajResult, N_ATOMS, ...

// One ensemble job. The geometry/potential are fixed in nnpmm.h; what varies
// per RUN is how many members, the integration timestep/length, and the size of
// the active-learning perturbation applied to the link atom.
struct EnsembleConfig {
    int    M = 0;        // number of trajectories (ensemble members)
    double dt = 0.0;     // velocity-Verlet timestep (time units)
    int    steps = 0;    // steps per trajectory (run length = steps*dt)
    double amp = 0.0;    // max link-atom perturbation (+/- amp across members)
};

// Number of ensemble members (kept as a function to mirror flagship style).
// NNPMM_HD so the GPU kernel can call it for its bounds check too (host+device).
NNPMM_HD inline int ensemble_size(const EnsembleConfig& c) { return c.M; }

// Load an EnsembleConfig from the text sample (see data/README.md):
//   "M dt steps amp"
EnsembleConfig load_ensemble(const std::string& path);

// CPU reference: run every trajectory serially. `results` is sized to M. This is
// the trusted baseline; the GPU ensemble must agree with it within tolerance
// (and, because both call run_trajectory() from nnpmm.h, the math is identical).
void integrate_cpu(const EnsembleConfig& c, std::vector<TrajResult>& results);
