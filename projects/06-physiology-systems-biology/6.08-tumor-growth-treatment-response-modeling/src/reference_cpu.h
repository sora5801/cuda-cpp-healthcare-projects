// ===========================================================================
// src/reference_cpu.h  --  Tumor params, field init, and the CPU reference
// ---------------------------------------------------------------------------
// Project 6.8 : Tumor Growth & Treatment-Response Modeling
//
// Pure C++ (no CUDA). The per-cell physics lives in tumor.h; kernels.cu reuses
// TumorParams and calls the same tumor_grow_update / tumor_treat_update, so the
// GPU reproduces this reference. The CPU path here is the trusted baseline that
// main.cu checks the GPU against (within the tolerance discussed in THEORY).
//
// READ THIS AFTER: tumor.h. READ THIS BEFORE: reference_cpu.cpp, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "tumor.h"   // TumorParams, tumor_grow_update, tumor_treat_update, lq_survival

// Load TumorParams from the whitespace-separated text format (see data/README.md):
//   nx ny dx D rho dt steps alpha beta dose n_fractions fx_interval seed_radius seed_u
// Throws std::runtime_error on a missing file or invalid/unstable parameters.
TumorParams load_tumor(const std::string& path);

// Build the initial density field u(x,y): 0 everywhere except a solid disc of
// radius `seed_radius` (in mm) at the grid centre, set to `seed_u`. This is the
// small starting tumor whose growth and treatment we simulate. `u` is sized
// nx*ny (row-major).
void init_field(const TumorParams& P, std::vector<double>& u);

// CPU reference simulation. Advances `u` in place for `steps` timesteps of
// Fisher-KPP growth (double-buffered explicit Euler), applying an LQ treatment
// fraction whenever the step index lands on the schedule. This is the exact
// serial twin of simulate_gpu(); the two final fields must agree within
// tolerance. Returns nothing; `u` holds the final density.
void simulate_cpu(const TumorParams& P, std::vector<double>& u);

// Shared schedule helper (host + kernels can both ask "is step s a fraction?").
// Returns true when timestep `s` (0-based, evaluated BEFORE the growth update of
// that step) should deliver a radiotherapy fraction. Fractions are delivered at
// steps fx_interval, 2*fx_interval, ... up to n_fractions of them. Defined here
// (not tumor.h) because only the time-loop needs it.
inline bool is_fraction_step(const TumorParams& P, int s) {
    if (P.n_fractions <= 0 || P.fx_interval <= 0) return false;
    if (s == 0) return false;                       // no dose at t=0 (seed only)
    if (s % P.fx_interval != 0) return false;       // only on interval boundaries
    const int fraction_number = s / P.fx_interval;  // 1st, 2nd, ... fraction
    return fraction_number <= P.n_fractions;        // stop after n_fractions
}
