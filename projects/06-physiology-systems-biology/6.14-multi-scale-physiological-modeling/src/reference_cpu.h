// ===========================================================================
// src/reference_cpu.h  --  Cable config loader + CPU reference simulation
// ---------------------------------------------------------------------------
// Project 6.14 : Multi-Scale Physiological Modeling
//
// WHY A SEPARATE PURE-C++ HEADER
//   reference_cpu.cpp is compiled by the plain host compiler (cl.exe / g++) and
//   must NOT see any CUDA/__global__ syntax, so its prototypes cannot live in
//   kernels.cuh. main.cu (nvcc) and reference_cpu.cpp (host) both include THIS
//   header so they agree on the config struct, the result struct, and the
//   function signatures. The actual per-node physics is in multiscale.h.
//
// WHAT THE CPU REFERENCE COMPUTES  (the trusted baseline, CLAUDE.md section 5)
//   The SAME monodomain reaction-diffusion propagation as the GPU, but done
//   serially: loop over global steps, and within each step loop over all nodes
//   for the reaction sub-step and the diffusion sub-step. Because both paths
//   call the shared __host__ __device__ routines in multiscale.h, the CPU and
//   GPU produce matching results (verified in main.cu within a documented,
//   physically-negligible tolerance -- see THEORY.md).
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "multiscale.h"   // CableConfig, FhnParams, react/diffusion routines

// ---------------------------------------------------------------------------
// CableResult -- the deterministic, physically-meaningful outputs the analysis
//   reports (and that main.cu verifies GPU-vs-CPU). Chosen so the numbers have
//   a clear physiological meaning and recover a KNOWN answer (a traveling wave):
//
//   activation_time[i] : the time at which node i first crosses the excitation
//       threshold (v >= 0.5). A propagating AP makes this INCREASE with i --
//       an activation "map". Sentinel -1 means the node never activated.
//   v_final[i], w_final[i] : the (v,w) state at the end of the run (used for a
//       full field-level GPU==CPU comparison, not just the summary metrics).
//   n_activated : how many nodes ever activated (the wave's reach).
//   conduction_velocity : distance/time of the wave over the activated span
//       (space units / time unit) -- the headline physiological quantity, the
//       cardiac analogue of "how fast the heartbeat spreads".
// ---------------------------------------------------------------------------
struct CableResult {
    std::vector<double> activation_time;   // [n] first-crossing time (or -1)
    std::vector<double> v_final;           // [n] excitation variable at t_end
    std::vector<double> w_final;           // [n] recovery variable at t_end
    int    n_activated = 0;                // number of nodes that activated
    double conduction_velocity = 0.0;      // measured wave speed (space/time)
};

// Load a CableConfig from the text sample (layout documented in data/README.md):
//   n dx dt steps stim_nodes a eps b D
// Throws std::runtime_error on a missing/short/invalid file so demos fail loudly.
CableConfig load_cable(const std::string& path);

// CPU reference: run the full split-step monodomain simulation serially and
//   fill `out`. This is the baseline the GPU kernel is checked against.
void simulate_cpu(const CableConfig& c, CableResult& out);

// Derive the conduction velocity + activation count from an activation map.
//   Shared by the CPU and GPU post-processing so the reported metric is defined
//   in exactly one place (declared here, defined in reference_cpu.cpp, and also
//   called from main.cu after the GPU run).
void summarize_activation(const CableConfig& c,
                          const std::vector<double>& activation_time,
                          int& n_activated, double& conduction_velocity);
