// ===========================================================================
// src/reference_cpu.h  --  System loader + CPU reference MD driver
// ---------------------------------------------------------------------------
// Project 1.1 : Molecular Dynamics Engine  (reduced-scope teaching version)
//
// ROLE IN THE PROJECT
//   Declares (a) how a simulation is loaded from the tiny sample file, and (b)
//   the CPU reference integrator -- the trusted, serial, easy-to-read baseline
//   that the GPU result is checked against. The per-pair physics and the Verlet
//   step live in md.h (shared host+device); this header is pure C++ so it is safe
//   to include from BOTH reference_cpu.cpp (host compiler) and kernels.cu (nvcc).
//
//   The CPU and GPU produce the SAME observable (an MdResult). main.cu runs both
//   and asserts they agree to a documented tolerance.
//
// READ THIS AFTER: md.h.  READ THIS BEFORE: reference_cpu.cpp, kernels.cuh.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "md.h"   // Vec3, SimParams, lj_pair_force, kinetic_energy_one, ...

// ---------------------------------------------------------------------------
// MdSystem: one fully-specified simulation = parameters + initial state.
//   pos / vel are length-n arrays of Vec3 (atom positions and velocities at t=0).
//   They are std::vector so the host owns them; the GPU path copies them to the
//   device. Stored as array-of-Vec3 here for clarity; THEORY discusses why a real
//   engine prefers structure-of-arrays (memory coalescing on the GPU).
// ---------------------------------------------------------------------------
struct MdSystem {
    SimParams         params;   // n, box, dt, steps, eps, sigma, rcut, mass
    std::vector<Vec3> pos;      // initial positions [n] (reduced length units)
    std::vector<Vec3> vel;      // initial velocities [n] (reduced velocity units)
};

// ---------------------------------------------------------------------------
// MdResult: the deterministic OBSERVABLES we report and verify. We deliberately
//   reduce a whole trajectory to a few scalars so CPU and GPU can be compared
//   exactly and the demo's stdout is stable:
//     E0           : total energy at t=0  (kinetic + potential)
//     E_final      : total energy after `steps` steps (should ~= E0: conservation)
//     max_drift    : largest |E(t) - E0| seen along the run (energy-drift metric)
//     pos_checksum : sum over atoms of (x+y+z) of the FINAL wrapped positions --
//                    a single number fingerprinting the final configuration, so
//                    any divergence between CPU and GPU trajectories shows here.
//     T_final      : instantaneous temperature at the end (from kinetic energy).
// ---------------------------------------------------------------------------
struct MdResult {
    double E0           = 0.0;
    double E_final      = 0.0;
    double max_drift    = 0.0;
    double pos_checksum = 0.0;
    double T_final      = 0.0;
};

// Load an MdSystem from the sample text format (see data/README.md):
//   line 1: n box dt steps eps sigma rcut mass
//   next n lines: x y z vx vy vz     (one atom per line)
// Throws std::runtime_error if the file is missing or malformed.
MdSystem load_system(const std::string& path);

// Generate the built-in synthetic system (used when no file is supplied) so the
// program always has something to run. Deterministic: same atoms every call.
MdSystem make_default_system();

// ---------------------------------------------------------------------------
// integrate_cpu: the REFERENCE molecular-dynamics driver (serial, O(steps*N^2)).
//   Runs the full velocity-Verlet loop on the host, computing all-pairs LJ forces
//   each step via the shared md.h physics, and returns the MdResult observables.
//   This is the baseline the GPU kernel must reproduce. The input system is not
//   mutated (it copies pos/vel internally) so the GPU path starts identically.
// ---------------------------------------------------------------------------
MdResult integrate_cpu(const MdSystem& sys);

// ---------------------------------------------------------------------------
// total_energy_cpu: helper computing (kinetic + potential) energy of a state.
//   Used inside the driver for the energy-drift diagnostic and exposed so main.cu
//   can report E0. Potential energy sums lj_pair_force's u over each unordered
//   pair exactly once (i < j).
// ---------------------------------------------------------------------------
double total_energy_cpu(const SimParams& p,
                        const std::vector<Vec3>& pos,
                        const std::vector<Vec3>& vel);
