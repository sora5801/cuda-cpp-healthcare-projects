// ===========================================================================
// src/reference_cpu.h  --  System definition + CPU reference simulation
// ---------------------------------------------------------------------------
// Project 2.30 : Protein Solubility & Phase Separation Simulation
//
// WHAT THIS HEADER DECLARES
//   * System          -- the full simulation state (params + per-bead arrays).
//   * SimSummary      -- the small set of DETERMINISTIC scalars we report+verify.
//   * load_system()   -- read the committed sample into a System.
//   * run_cpu()       -- the serial velocity-Verlet reference simulation.
//   * order_params()  -- the phase-separation diagnostics that summarize a state.
//
//   This header is pure C++ (no CUDA), so kernels.cu can reuse `System` and the
//   analysis without dragging CUDA types into the host compiler. The actual
//   per-pair physics lives in hps_model.h (shared host+device); the CPU
//   reference just loops it serially, exactly as the GPU kernel does in
//   parallel -- which is why their trajectories agree (see THEORY.md "verify").
//
// READ THIS AFTER: hps_model.h.   READ THIS BEFORE: reference_cpu.cpp, main.cu.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "hps_model.h"   // SimParams + the shared force/energy formulas

// ---------------------------------------------------------------------------
// System: everything needed to advance the simulation and analyze it.
//   Positions/velocities are stored as flat structure-of-arrays (x[],y[],z[]):
//   this is the GPU-friendly layout because thread t reads x[t] from one
//   contiguous stream (coalesced global-memory access) instead of striding
//   through an array of structs. We keep the same layout on the CPU so the two
//   paths share indexing and there is no transpose to get wrong.
// ---------------------------------------------------------------------------
struct System {
    SimParams p{};                  // box, force-field, and integrator parameters

    // Per-bead state (length p.n_beads each). Double precision throughout.
    std::vector<double> x, y, z;    // positions (reduced length units)
    std::vector<double> vx, vy, vz; // velocities (reduced length / time)
    std::vector<double> lambda;     // per-residue stickiness in [0,1]
    std::vector<int>    chain_id;   // which chain each bead belongs to (0..n_chains-1)

    int n() const { return p.n_beads; }
};

// ---------------------------------------------------------------------------
// SimSummary: the small, DETERMINISTIC set of numbers we print and verify.
//   Rather than diff millions of coordinates, we summarize the final state with
//   physically meaningful scalars (energy, a phase order parameter, a coordinate
//   checksum). main.cu compares the CPU and GPU summaries field by field.
// ---------------------------------------------------------------------------
struct SimSummary {
    double potential;          // total potential energy at the final step
    double kinetic;            // total kinetic energy at the final step
    double pos_checksum;       // sum over beads of folded (x+y+z): a trajectory fingerprint
    double max_local_density;  // densest local neighbourhood (the condensate signal)
    double mean_local_density; // average neighbourhood occupancy (dilute baseline)
    int    n_condensed;        // beads whose local density exceeds a threshold
};

// Load a System from the sample text file (format documented in data/README.md).
// Throws std::runtime_error on a missing/malformed file so demos fail loudly.
System load_system(const std::string& path);

// ---------------------------------------------------------------------------
// run_cpu: the serial reference simulation.
//   Advances a COPY of the System by p.n_steps velocity-Verlet steps using the
//   shared hps_model.h forces, then fills `out` with the SimSummary.
//   Determinism: forces are summed in a FIXED bead order (j = 0..N-1), the same
//   order the GPU kernel uses, so the two FP64 trajectories stay bit-close.
// ---------------------------------------------------------------------------
void run_cpu(System sys, SimSummary& out);

// ---------------------------------------------------------------------------
// order_params: compute the phase-separation diagnostics from a set of
//   positions. "Local density" of a bead = how many other beads sit within a
//   probe radius (here r_cut). A condensate shows up as a subset of beads with
//   high local density (the dense droplet) against a low-density background (the
//   dilute phase). This is the catalog's "order-parameter clustering for phase
//   detection", in its simplest interpretable form. Deterministic: integer
//   neighbour counts summed in index order.
// ---------------------------------------------------------------------------
void order_params(const SimParams& p,
                  const std::vector<double>& x,
                  const std::vector<double>& y,
                  const std::vector<double>& z,
                  double& max_local_density,
                  double& mean_local_density,
                  int& n_condensed);
