// ===========================================================================
// src/reference_cpu.h  --  System builder + CPU MD reference (pure C++)
// ---------------------------------------------------------------------------
// Project 2.19 : Membrane Protein Simulation   (reduced-scope teaching version)
//
// Pure C++ (no CUDA): compiled by cl.exe / g++. The per-bead and per-pair
// PHYSICS lives in membrane.h (shared with the GPU). This header declares:
//   * load_params()  -- read the one-line SimParams from the sample file,
//   * build_system() -- lay out the initial bilayer + protein bead positions,
//                       velocities, masses, and per-bead types,
//   * simulate_cpu() -- the trusted serial velocity-Verlet + Langevin loop the
//                       GPU result is verified against.
//   * Observables    -- bilayer_thickness(), total_potential_energy() for the report.
//
// kernels.cu reuses SimParams, Vec3, and the System layout, so CPU and GPU
// integrate the SAME particles with the SAME math and must agree.
//
// READ THIS AFTER: membrane.h.  READ THIS BEFORE: reference_cpu.cpp, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "membrane.h"   // Vec3, SimParams, BeadType, all force/integrator math

// ---------------------------------------------------------------------------
// System: the flat arrays the simulation operates on. We use Structure-of-
// Arrays (separate vectors) rather than Array-of-Structs because the GPU reads
// them COALESCED -- thread i touching pos[i], vel[i] gives neighbouring threads
// neighbouring addresses, the access pattern global memory likes best.
// (THEORY "GPU mapping" explains the coalescing win.)
// ---------------------------------------------------------------------------
struct System {
    std::vector<Vec3>   pos;     // [n_beads] positions
    std::vector<Vec3>   vel;     // [n_beads] velocities
    std::vector<double> inv_mass;// [n_beads] 1/mass (0 would pin a bead; none here)
    std::vector<double> mass;    // [n_beads] mass (Langevin needs m, not 1/m)
    std::vector<int>    type;    // [n_beads] BeadType (HEAD/TAIL/PROT)
    // Bonds as index pairs (i,j); lipid head-tail and tail-tail springs.
    std::vector<int>    bond_i;  // [n_bonds]
    std::vector<int>    bond_j;  // [n_bonds]
};

// Load SimParams from the whitespace text format documented in data/README.md.
// Throws std::runtime_error on a missing file or malformed parameters so demos
// fail loudly rather than silently simulating garbage.
SimParams load_params(const std::string& path);

// Build the initial configuration deterministically from P:
//   * a square raster of lipids split into two leaflets (upper/lower), each
//     lipid = HEAD bead near the water side + two TAIL beads pointing toward
//     the bilayer midplane, so tails of opposite leaflets meet (a real bilayer),
//   * `n_prot` PROTEIN beads stacked as a short column through the membrane core,
//   * small deterministic initial velocities (seeded) so there is motion to
//     thermostat but the run is still reproducible.
// Fills every field of `sys`. No randomness beyond the seeded velocities.
void build_system(const SimParams& P, System& sys);

// CPU reference: advance the system `P.steps` velocity-Verlet steps with a
// Langevin thermostat, in place. This is the trusted baseline; the GPU must
// reproduce sys.pos / sys.vel within tolerance. O(steps * n_beads^2).
void simulate_cpu(const SimParams& P, System& sys);

// ---- Observables (shared by CPU and GPU paths; deterministic) --------------

// Bilayer thickness = (mean z of upper-leaflet HEAD beads) - (mean z of
// lower-leaflet HEAD beads). The canonical "is the membrane intact?" number.
double bilayer_thickness(const SimParams& P, const System& sys);

// Total potential energy (LJ + bonds) of the configuration -- a second,
// physics-level diagnostic that CPU and GPU should also agree on.
double total_potential_energy(const SimParams& P, const System& sys);
