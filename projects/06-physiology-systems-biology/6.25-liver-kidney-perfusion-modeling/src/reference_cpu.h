// ===========================================================================
// src/reference_cpu.h  --  Lobule config + CPU reference perfusion solve
// ---------------------------------------------------------------------------
// Project 6.25 : Liver & Kidney Perfusion Modeling
//
// A LOBULE is modeled as an ENSEMBLE of `nsin` sinusoids that share the same
// physical constants (SinusoidParams) but each carry a different inlet blood
// VELOCITY. Perfusion is heterogeneous: real lobules have a spread of flow
// rates, so we sweep velocity linearly from v_lo (slow, well-cleared) to v_hi
// (fast, poorly-cleared). Each sinusoid is an INDEPENDENT 1-D transport-reaction
// ODE solve (perfusion.h), so member index -> one GPU thread in kernels.cu.
//
// This header holds the config, the (index -> velocity) mapping, the loader, and
// the CPU reference declaration. The actual transport physics/RK4 is in
// perfusion.h. Pure C++; kernels.cu reuses LobuleConfig directly.
//
// WHY A SEPARATE HEADER: the CPU reference (reference_cpu.cpp) is compiled by the
// plain C++ compiler and must NOT see any CUDA/__global__ syntax, so its config
// and prototype live here, not in kernels.cuh. Both main.cu and reference_cpu.cpp
// include this pure-C++ header so they agree on the types and signatures.
//
// READ THIS AFTER: perfusion.h. READ BEFORE: kernels.cuh, reference_cpu.cpp.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "perfusion.h"   // PERF_HD, SinusoidParams, SinusoidResult, integrate_sinusoid

// ---------------------------------------------------------------------------
// LobuleConfig : one whole-lobule job = fixed sinusoid physics + a velocity
//   sweep across the `nsin` parallel sinusoids. This is the input the demo
//   loads from data/sample/lobule.txt (see data/README.md for the field order).
// ---------------------------------------------------------------------------
struct LobuleConfig {
    SinusoidParams p;       // physical constants shared by all sinusoids in the lobule
    int    nsin = 0;        // number of parallel sinusoids (ensemble members)
    double v_lo = 0.0;      // slowest inlet blood velocity (mm/s) -> longest residence, most cleared
    double v_hi = 0.0;      // fastest inlet blood velocity (mm/s) -> shortest residence, least cleared
};

// Number of ensemble members (sinusoids) in the lobule.
PERF_HD inline int lobule_size(const LobuleConfig& c) { return c.nsin; }

// ---------------------------------------------------------------------------
// sinusoid_velocity: map a flat sinusoid index to its inlet blood velocity on
//   the linear sweep. idx in [0, nsin) -> v = v_lo + idx/(nsin-1)*(v_hi-v_lo).
//   Shared host+device so the kernel and the CPU reference pick identical v.
// ---------------------------------------------------------------------------
PERF_HD inline double sinusoid_velocity(const LobuleConfig& c, int idx) {
    return (c.nsin > 1) ? c.v_lo + (c.v_hi - c.v_lo) * idx / (c.nsin - 1) : c.v_lo;
}

// Load a LobuleConfig from the whitespace text format documented in data/README.md:
//   "L C_in Km Vmax_pp Vmax_cl nseg   nsin v_lo v_hi"
// Throws std::runtime_error on a missing file or invalid (non-positive) values.
LobuleConfig load_lobule(const std::string& path);

// CPU reference: integrate every sinusoid serially into `results` (sized nsin).
//   The trusted baseline the GPU ensemble is checked against -- identical RK4
//   (perfusion.h) means the numbers match to round-off.
void integrate_cpu(const LobuleConfig& c, std::vector<SinusoidResult>& results);
