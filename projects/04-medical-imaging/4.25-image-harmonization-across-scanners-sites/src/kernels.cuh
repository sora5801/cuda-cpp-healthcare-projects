// ===========================================================================
// src/kernels.cuh  --  GPU ComBat interface (declarations + the teaching idea)
// ---------------------------------------------------------------------------
// Project 4.25 : Image Harmonization Across Scanners/Sites
//
// THE BIG IDEA (pattern: ENSEMBLE OF INDEPENDENT PER-FEATURE SOLVES)
//   ComBat harmonizes every FEATURE independently: fit a small regression,
//   estimate this feature's per-batch location/scale, shrink them toward the
//   panel-wide empirical-Bayes priors, and subtract the batch signature. Because
//   feature p never touches feature q, the natural GPU mapping is ONE THREAD PER
//   FEATURE -- thread p runs the entire ComBat pipeline for row p in registers.
//   No shared memory, no atomics, no cross-thread reduction: the same "ensemble
//   of tiny solves" shape as the ODE-ensemble flagships (9.02 SEIR, 13.02 PBPK).
//
//   The per-feature math is the SHARED __host__ __device__ core cb_harmonize_
//   feature() in combat.h, so the GPU thread and the CPU reference execute
//   byte-for-byte identical arithmetic (PATTERNS.md §2) -> exact verification.
//
//   The design matrix and the EB priors are computed ONCE on the host (they are
//   the same across features, and the prior fit is a cheap across-feature reduce)
//   and uploaded read-only; kernels.cu just launches the per-feature kernel.
//
// READ THIS AFTER: combat.h, reference_cpu.h, util/cuda_check.cuh, util/timer.cuh.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // Dataset + build_design + estimate_priors (pure C++)

// combat_gpu: harmonize the whole [P x N] feature table on the GPU.
//   Inputs mirror combat_cpu(): the dataset, the shared design matrix, the EB
//   priors, and the per-batch sample counts (all host vectors; copied to device
//   inside). Fills `out` [P*N] with the harmonized table and returns the GPU
//   kernel time via kernel_ms (CUDA-event measured, kernel only -- not copies).
void combat_gpu(const Dataset& d, const std::vector<double>& design,
                const std::vector<double>& gamma_bar, const std::vector<double>& tau2,
                const std::vector<double>& a_prior,   const std::vector<double>& b_prior,
                const std::vector<int>&    batch_n,
                std::vector<double>& out, float* kernel_ms);
