// ===========================================================================
// src/kernels.cuh  --  GPU ensemble-integration interface (whole-heart twins)
// ---------------------------------------------------------------------------
// Project 6.2 : Whole-Heart Digital Twin   (REDUCED-SCOPE TEACHING VERSION)
//
// THE BIG IDEA (the ENSEMBLE-ODE GPU pattern)
//   A digital twin is calibrated by running the SAME forward heart model for
//   many parameter samples (here: a contractility sweep) and comparing each
//   output to a clinical target. Each forward solve is sequential in TIME but
//   completely INDEPENDENT of the other members, so we give each virtual heart
//   its own GPU thread: the thread runs the full multi-beat RK4 loop in
//   registers and writes one TwinResult. This is the "thousands of forward
//   simulations for the inference step" the catalog deep-dive calls out, mapped
//   to the GPU the natural way (PATTERNS.md section 1, ensemble RK4).
//
//   The heart physics + RK4 are shared with the CPU (heart.h), so the GPU
//   results match the reference to round-off. kernels.cu defines the kernel.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, heart.h, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // EnsembleConfig, TwinResult (pure C++, safe in .cu)

// Device kernel: thread `idx` simulates ensemble member idx (its own virtual
// heart) and writes its TwinResult. Reads its E_max from the sweep via
// member_params(). One thread == one heart == no inter-thread communication.
__global__ void ensemble_kernel(EnsembleConfig c, TwinResult* __restrict__ out);

// Host wrapper: launch one thread per member, copy the results back, and time
// just the kernel with CUDA events. `kernel_ms` receives the GPU time (ms).
void integrate_gpu(const EnsembleConfig& c, std::vector<TwinResult>& results, float* kernel_ms);
