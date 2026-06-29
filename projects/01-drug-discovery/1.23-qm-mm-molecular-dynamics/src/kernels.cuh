// ===========================================================================
// src/kernels.cuh  --  GPU ensemble-integration interface
// ---------------------------------------------------------------------------
// Project 1.23 : QM/MM Molecular Dynamics   (reduced-scope teaching version)
//
// THE BIG IDEA (ENSEMBLE QM/MM TRAJECTORIES, one thread per trajectory)
//   A QM/MM simulation is sequential in time -- each MD step needs the previous
//   step's geometry to evaluate the next quantum force. So a SINGLE trajectory
//   is not data-parallel. What IS parallel is running MANY trajectories at once:
//   a sweep over the MM electrostatic-embedding field and the initial proton
//   position. Each (field, x0) pair is an independent run, so we give each its
//   own GPU thread; the thread executes the full velocity-Verlet loop in
//   registers and writes one TrajResult. This is exactly the ensemble pattern
//   used by the 9.02 (SEIR) and 13.02 (PBPK) flagships (PATTERNS.md §1).
//
//   Why this is the right teaching mapping for QM/MM: real reactive-event
//   sampling (e.g. proton-transfer free energies, committor analysis) launches
//   THOUSANDS of trajectories from perturbed initial conditions -- an ensemble
//   over starting states is precisely how production QM/MM gathers statistics.
//
//   The per-step physics + Verlet integrator are shared with the CPU via qmmm.h,
//   so the GPU results match the reference to round-off. kernels.cu defines the
//   kernel and the host wrapper.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, qmmm.h, reference_cpu.h.
// Then read kernels.cu.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // EnsembleConfig, member_params (pure C++, safe in .cu)
#include "qmmm.h"            // qmmm::TrajResult, integrate_trajectory

// ---- Device kernel -------------------------------------------------------
// ensemble_kernel: thread `idx` integrates ensemble member idx and writes its
//   TrajResult. It reads its (field, x0) from the sweep via member_params(), then
//   runs the entire Verlet time loop (qmmm::integrate_trajectory) in registers.
//   Launch config (set in integrate_gpu): grid = ceil(M / block), block = 128.
//   Thread-to-data map: idx = blockIdx.x*blockDim.x + threadIdx.x owns member idx.
//   Memory: reads the small EnsembleConfig (passed by value), writes one struct to
//   global `out[idx]`; no shared memory or atomics (members are independent).
__global__ void ensemble_kernel(EnsembleConfig c, qmmm::TrajResult* __restrict__ out);

// ---- Host wrapper --------------------------------------------------------
// integrate_gpu: launch one thread per member, copy the results back, and report
//   the measured KERNEL time (CUDA events) via *kernel_ms. main.cu calls exactly
//   this; all CUDA bookkeeping (malloc / launch / memcpy / free) is hidden here.
//     c         : the ensemble configuration (sweep grid + integration settings)
//     results   : host output, resized to nf*nx (output parameter)
//     kernel_ms : out-param, milliseconds spent in the kernel itself (not copies)
void integrate_gpu(const EnsembleConfig& c, std::vector<qmmm::TrajResult>& results,
                   float* kernel_ms);
