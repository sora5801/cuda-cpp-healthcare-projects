// ===========================================================================
// src/reference_cpu.h  --  Loader + CPU coarse-grained MD reference
// ---------------------------------------------------------------------------
// Project 2.5 : Coarse-Grained / MARTINI Simulation
//
// Pure C++ (no CUDA). The per-pair physics and the velocity-Verlet update live
// in martini.h; kernels.cu reuses MdParams, Vec3, and those functions. The CPU
// reference runs the IDENTICAL MD steps as the GPU, summing pair forces in the
// same index order, so the final positions/velocities match within a tight
// double-precision tolerance (see THEORY section 6). This header is included by
// both reference_cpu.cpp (host compiler) and main.cu / kernels.cu (nvcc), so it
// must stay free of any __global__ / CUDA-only syntax.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "martini.h"   // Vec3, MdParams, compute_force_on, verlet_* helpers

// A complete CG system: parameters + per-bead positions, velocities, and types.
// Bundling them keeps the loader, the CPU reference, and the GPU wrapper all
// speaking the same data layout (a Structure-of-Arrays would be faster on the
// GPU but Array-of-Structs Vec3 is clearer for teaching -- see THEORY section 4).
struct System {
    MdParams          P;     // simulation constants (box, dt, eps matrix, ...)
    std::vector<Vec3> pos;   // [n] bead positions  (nm)
    std::vector<Vec3> vel;   // [n] bead velocities (nm / time-unit)
    std::vector<int>  type;  // [n] bead types: 0 = C (apolar), 1 = P (polar)
};

// Load a System from the committed text format (see data/README.md):
//   header line:  n box dt steps rcut mass sigma epsCC epsCP epsPP
//   then n lines: x y z vx vy vz type
// Throws std::runtime_error on a missing/malformed file.
System load_system(const std::string& path);

// CPU reference: advance the system `steps` velocity-Verlet steps IN PLACE.
// This is the trusted baseline the GPU result is checked against. Heavily
// commented in reference_cpu.cpp; the physics itself is in martini.h.
void simulate_cpu(System& sys);

// --- Diagnostics shared by main.cu for both the CPU and GPU final states ---

// total_energy: kinetic + Lennard-Jones potential of the whole system.
//   A (nearly) conserved quantity -- a good sanity check on the integrator.
double total_energy(const System& sys);

// cp_separation: distance between the C-bead centroid and the P-bead centroid.
//   It grows as oil-like C beads demix from water-like P beads -- the physical
//   signal this demo is built to show. Deterministic given the bead positions.
double cp_separation(const System& sys);
