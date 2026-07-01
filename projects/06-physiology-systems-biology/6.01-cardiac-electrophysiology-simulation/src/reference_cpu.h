// ===========================================================================
// src/reference_cpu.h  --  Monodomain parameters + serial CPU reference solver
// ---------------------------------------------------------------------------
// Project 6.1 : Cardiac Electrophysiology Simulation
//
// Pure C++ (no CUDA). kernels.cu reuses MonodomainParams. The actual per-cell
// physics (FitzHugh-Nagumo reaction + diffusion stencil) lives in the shared
// cardiac_cell.h so CPU and GPU compute byte-for-byte identical results.
//
// READ THIS AFTER: cardiac_cell.h (the physics). READ BEFORE: reference_cpu.cpp.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "cardiac_cell.h"   // MonodomainParams + shared host/device update

// Load a MonodomainParams from the whitespace-separated sample format
// (documented in data/README.md), in this exact field order:
//
//   nx ny steps dt dx D a eps b  stim_x0 stim_y0 stim_w stim_h stim_v
//
// Throws std::runtime_error if the file is missing or malformed, so the demo
// fails loudly instead of silently simulating garbage.
MonodomainParams load_monodomain(const std::string& path);

// Initialise the tissue: V = 0 (rest) and w = 0 everywhere, then clamp the S1
// stimulus patch to stim_v. Fills V and w (each size nx*ny). Shared by BOTH the
// CPU and GPU paths so they start from an identical state.
void init_state(const MonodomainParams& p,
                std::vector<double>& V, std::vector<double>& w);

// CPU reference: run `steps` operator-split timesteps (reaction half-step, then
// diffusion half-step), starting from init_state(), and return the FINAL
// voltage field V and recovery field w (each size nx*ny). This is the trusted
// baseline the GPU result is checked against.
void monodomain_cpu(const MonodomainParams& p,
                    std::vector<double>& V_final, std::vector<double>& w_final);
