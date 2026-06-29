// ===========================================================================
// src/reference_cpu.h  --  Ensemble config + CPU reference integration
// ---------------------------------------------------------------------------
// Project 1.23 : QM/MM Molecular Dynamics   (reduced-scope teaching version)
//
// The ensemble is a 2-D parameter SWEEP of QM/MM trajectories:
//   nf values of the MM electrostatic-embedding FIELD  x  nx values of the
//   INITIAL proton displacement  =  nf*nx independent QM/MM runs.
// Each run is a full velocity-Verlet integration whose force comes from a 2x2
// quantum solve at every step (see qmmm.h). The config + the flat-index ->
// (field, x0) mapping live here (shared host+device so the GPU kernel reuses
// them); the actual physics/Verlet is in qmmm.h. This file is pure C++ so the
// host compiler builds reference_cpu.cpp and nvcc reuses EnsembleConfig in
// kernels.cu.
//
// READ THIS AFTER: qmmm.h.   READ THIS BEFORE: reference_cpu.cpp, kernels.cuh.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "qmmm.h"   // QMMM_HD, integrate_trajectory, TrajResult

// One ensemble job: fixed integration settings + the two swept parameter ranges.
struct EnsembleConfig {
    double dt = 0.0;        // velocity-Verlet timestep (model time units)
    int    steps = 0;       // number of Verlet steps per trajectory
    double v0 = 0.0;        // initial proton velocity (shared by all members)
    int    nf = 0, nx = 0;  // sweep grid: nf field values x nx initial-position values
    double field_lo = 0.0, field_hi = 0.0;  // MM embedding-field range (energy/length)
    double x0_lo = 0.0,    x0_hi = 0.0;      // initial proton displacement range (length)
};

// Number of ensemble members (trajectories).
QMMM_HD inline int ensemble_size(const EnsembleConfig& c) { return c.nf * c.nx; }

// Map a flat member index to its (field, x0) on the sweep grid.
//   idx = a*nx + b  ->  field from row a (0..nf-1), x0 from column b (0..nx-1).
//   Linear interpolation across [lo, hi]; a single-value axis pins to its lo.
QMMM_HD inline void member_params(const EnsembleConfig& c, int idx,
                                  double& field, double& x0) {
    const int a = idx / c.nx;   // field index (row)
    const int b = idx % c.nx;   // initial-position index (column)
    field = (c.nf > 1) ? c.field_lo + (c.field_hi - c.field_lo) * a / (c.nf - 1) : c.field_lo;
    x0    = (c.nx > 1) ? c.x0_lo    + (c.x0_hi    - c.x0_lo)    * b / (c.nx - 1) : c.x0_lo;
}

// Load an EnsembleConfig from the text format (see data/README.md):
//   "dt steps v0 nf nx field_lo field_hi x0_lo x0_hi"
EnsembleConfig load_ensemble(const std::string& path);

// CPU reference: integrate every member serially. `results` is sized to nf*nx.
// This is the trusted baseline the GPU ensemble is checked against; because both
// call qmmm::integrate_trajectory, the numbers agree to round-off.
void integrate_cpu(const EnsembleConfig& c, std::vector<qmmm::TrajResult>& results);
