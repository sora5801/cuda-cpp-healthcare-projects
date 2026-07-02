// ===========================================================================
// src/reference_cpu.h  --  Problem definition + CPU reference for the DFT sweep
// ---------------------------------------------------------------------------
// Project 6.19 : Defibrillation & High-Voltage Shock Simulation
//
// WHY A SEPARATE PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the plain C++ compiler and must NOT see any
//   CUDA/__global__ syntax, so its prototypes cannot live in kernels.cuh. Both
//   main.cu (nvcc) and reference_cpu.cpp (cl.exe) include THIS header, so they
//   agree on the data types and function signatures. The actual per-step physics
//   is in the shared defib.h (host+device), keeping CPU and GPU in lockstep.
//
// WHAT WE COMPUTE
//   A DEFIBRILLATION-THRESHOLD (DFT) sweep. We take a 1-D cardiac cable that
//   starts with a region of ongoing electrical activity, then apply a
//   defibrillation shock of a given amplitude and simulate forward. We repeat
//   this for a whole RANGE of shock amplitudes and record, for each, the
//   residual activity left in the tissue afterwards. The DFT is the smallest
//   amplitude that drives the residual activity below a success threshold --
//   i.e. the weakest shock that reliably terminates the arrhythmia.
//
//   Each amplitude's simulation is completely independent of the others, which
//   is exactly why the GPU helps: one thread runs one full cable simulation for
//   one shock amplitude (the "ensemble of trajectories" pattern -- see 9.02 /
//   13.02 in docs/PATTERNS.md section 1).
//
// READ THIS AFTER: defib.h. Then reference_cpu.cpp, kernels.cuh, main.cu.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "defib.h"   // FhnParams + shared host/device physics

// ---------------------------------------------------------------------------
// ShockSweep -- one complete "experiment": the shared cable/FHN parameters plus
// the list of shock amplitudes to test. Loaded from the tiny text sample.
// ---------------------------------------------------------------------------
struct ShockSweep {
    FhnParams           p;          // cable + FHN + shock-timing parameters
    std::vector<double> amps;       // shock amplitudes to sweep (dimensionless)
    double              success_thresh = 0.0;  // residual-activity level below
                                    //   which a shock counts as "defibrillated"
};

// ---------------------------------------------------------------------------
// load_sweep: parse the sample text file (format documented in data/README.md):
//   line 1: ncell nsteps dt dx D a eps gamma
//   line 2: initial_excited shock_start shock_len biphasic success_thresh
//   line 3: namp   a0 a1 a2 ...            (namp shock amplitudes follow)
// Throws std::runtime_error on any malformed input so demos fail loudly.
// ---------------------------------------------------------------------------
ShockSweep load_sweep(const std::string& path);

// ---------------------------------------------------------------------------
// simulate_one_cpu: run ONE full cable simulation for shock amplitude `amp` and
// return the post-shock residual-activity metric (activity_metric from defib.h).
//   This is the trusted serial baseline; the GPU kernel computes the same number
//   for the same amplitude and we assert they agree.
// ---------------------------------------------------------------------------
double simulate_one_cpu(const FhnParams& p, double amp);

// ---------------------------------------------------------------------------
// sweep_cpu: run simulate_one_cpu for every amplitude in `s.amps`, filling
// `residual` (resized to amps.size()) with the residual activity of each.
// ---------------------------------------------------------------------------
void sweep_cpu(const ShockSweep& s, std::vector<double>& residual);

// ---------------------------------------------------------------------------
// find_dft: given the swept amplitudes and their residual activities, return the
// INDEX of the smallest amplitude whose residual is below success_thresh (the
// defibrillation threshold), or -1 if no tested shock succeeded. Deterministic:
// amplitudes are assumed sorted ascending, and we return the first success.
// ---------------------------------------------------------------------------
int find_dft(const ShockSweep& s, const std::vector<double>& residual);
