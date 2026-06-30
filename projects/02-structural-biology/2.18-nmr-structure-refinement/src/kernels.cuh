// ===========================================================================
// src/kernels.cuh  --  GPU ensemble-annealing interface
// ---------------------------------------------------------------------------
// Project 2.18 : NMR Structure Refinement
//
// THE BIG IDEA (the "ensemble of independent annealers" pattern)
//   An NMR structure is determined by running HUNDREDS of independent simulated-
//   annealing trajectories from different random seeds and keeping the lowest-
//   energy ones. Every trajectory is independent, so we give each replica its OWN
//   GPU thread: the thread runs the full Monte-Carlo annealing loop (in per-thread
//   local memory) and writes one ReplicaResult. There is no inter-thread
//   communication -- pure embarrassing parallelism over replicas. This is the
//   same shape as the 9.02 SEIR and 13.02 PBPK ensembles (PATTERNS.md section 1),
//   but the per-thread loop is a Metropolis annealer (nmr_refine.h) rather than an
//   RK4 integrator.
//
//   The annealer (anneal_one) is shared with the CPU via nmr_refine.h, so a given
//   replica produces IDENTICAL numbers on host and device. kernels.cu defines the
//   kernel and the host launch wrapper.
//
//   Included only by .cu translation units (it contains a __global__ declaration,
//   so the plain C++ host compiler must never see it -- that is why the loader and
//   CPU reference live in the separate pure-C++ reference_cpu.h).
//
// READ THIS AFTER: nmr_refine.h, reference_cpu.h, util/cuda_check.cuh, util/timer.cuh.
// ===========================================================================
#pragma once

#include <vector>

#include "reference_cpu.h"   // RefineConfig, ReplicaResult (pure C++, safe in .cu)

// ---- Device kernel -------------------------------------------------------
// anneal_kernel: thread r anneals replica r and writes its ReplicaResult.
//   Declared here, defined in kernels.cu.
//   RefineConfig is passed BY VALUE so the whole job -- including the restraint
//   list -- travels in the kernel's parameter space (read by every thread with no
//   extra cudaMalloc). ReplicaResult* out has one slot per replica.
__global__ void anneal_kernel(RefineConfig c, ReplicaResult* __restrict__ out);

// ---- Host wrapper --------------------------------------------------------
// anneal_ensemble_gpu: launch one thread per replica, copy results back, and
//   report the measured KERNEL time (CUDA events) through *kernel_ms. This is the
//   GPU twin of anneal_ensemble_cpu(); main.cu runs both and compares them.
//
//   c          : the refinement job (chain, restraints, schedule)
//   results    : host output, resized to c.n_replicas (output parameter)
//   kernel_ms  : out-param, milliseconds spent in the kernel itself (not copies)
void anneal_ensemble_gpu(const RefineConfig& c,
                         std::vector<ReplicaResult>& results, float* kernel_ms);
