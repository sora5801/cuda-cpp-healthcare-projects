// ===========================================================================
// src/reference_cpu.h  --  Config loader + CPU reference for FEP/TI
// ---------------------------------------------------------------------------
// Project 1.5 : Free Energy Perturbation / Thermodynamic Integration
//
// The CPU reference is the trusted baseline the GPU is checked against. Because
// the actual physics + MC sampler live in alchemy.h (shared __host__ __device__),
// the "reference" here is just: load the config, then loop over lambda-windows
// calling run_chain() serially. The GPU kernel runs the SAME run_chain() per
// thread, so the two agree to round-off. Pure C++ (no CUDA), so kernels.cu can
// reuse AlchemyConfig safely.
//
// READ THIS AFTER: alchemy.h (the model/RNG/sampler).  Loader impl: reference_cpu.cpp.
// ===========================================================================
#pragma once

#include <string>
#include <vector>

#include "alchemy.h"   // AlchemyConfig, run_chain, ALCH_HD, n_windows

// Load an AlchemyConfig from the whitespace-separated text format used by
// data/sample/ (field order documented in data/README.md):
//   kA x0A kB x0B kT windows equil samples step x_init
// Throws std::runtime_error on a missing/short/invalid file so demos fail loudly.
AlchemyConfig load_config(const std::string& path);

// CPU reference: for each lambda-window, run one MC chain and store its estimate
// of < dU/dlambda >_lambda.  `dvals` is sized to n_windows(c). `accepted` (if
// non-null) receives the per-window accepted-move counts for an acceptance-rate
// diagnostic. This is the serial twin of the GPU kernel in kernels.cu.
void integrate_cpu(const AlchemyConfig& c,
                   std::vector<double>& dvals,
                   std::vector<long long>& accepted);
