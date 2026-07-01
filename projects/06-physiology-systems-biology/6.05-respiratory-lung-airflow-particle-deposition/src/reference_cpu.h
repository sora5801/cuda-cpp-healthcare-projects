// ===========================================================================
// src/reference_cpu.h  --  Problem definition + CPU deposition reference
// ---------------------------------------------------------------------------
// Project 6.5 : Respiratory / Lung Airflow & Particle Deposition
//
// The CPU reference tracks the SAME particle histories as the GPU (same RNG,
// same deposition physics from lung_physics.h), so the two per-generation
// deposition tallies must be identical. This header is pure C++ (no CUDA);
// kernels.cu reuses DepositionProblem and the Airway/Particle structs.
// ===========================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "lung_physics.h"   // Particle, Airway, Rng, track_particle (host+device)

// ---------------------------------------------------------------------------
// A complete deposition experiment read from the sample file:
//   * one monodisperse aerosol (diameter + density),
//   * the breathing flow rate (sets airway velocities),
//   * how many particle histories to launch, and the RNG seed.
// The airway geometry itself is BUILT deterministically from n_gen + flow_rate
// by build_airway() below (a scaled Weibel-A tree), so the tiny sample file
// stays human-readable -- see data/README.md for the field order.
// ---------------------------------------------------------------------------
struct DepositionProblem {
    // --- aerosol ---
    double d_p;        // particle aerodynamic diameter                     [m]
    double rho_p;      // particle mass density                            [kg/m^3]
    // --- airway / breathing ---
    int    n_gen;      // number of conducting-airway generations to model
    double flow_rate;  // inspiratory volumetric flow rate                 [m^3/s]
    // --- Monte Carlo ---
    uint64_t n_particles;  // number of particle histories to track
    uint64_t seed;         // base RNG seed (particle i uses stream (seed,i))
};

// Load a DepositionProblem from the one-line text format (see data/README.md):
//   "d_p_microns rho_p n_gen flow_L_per_min n_particles seed"
// (Convenience units in the file -- microns and L/min -- are converted to SI
// here so the sample is easy to read and edit.)
DepositionProblem load_problem(const std::string& path);

// Build the symmetric bifurcating airway tree for this problem. Generation g
// has 2^g parallel tubes; radius and length shrink geometrically with g
// (Weibel-A scaling), and continuity fixes the per-tube velocity from the total
// flow. Returns a fully-populated Airway (n_gen generations). Deterministic and
// pure -- shared by the CPU reference and the GPU host wrapper so both sides see
// identical geometry.
lung::Airway build_airway(const DepositionProblem& prob);

// CPU reference: track all n_particles histories serially and tally an INTEGER
// count per generation (index n_gen holds the "exhaled" count, so `tally` has
// n_gen+1 entries). Because the physics is bit-identical to the GPU and the
// tally is integer, this must equal the GPU result exactly.
void deposition_cpu(const DepositionProblem& prob, const lung::Airway& aw,
                    std::vector<uint64_t>& tally);
