// ===========================================================================
// src/kernels.cuh  --  GPU cohort-simulation interface
// ---------------------------------------------------------------------------
// Project 6.23 : Glucose-Insulin Dynamics & Artificial Pancreas
//
// THE BIG IDEA (pattern: ENSEMBLE ODE INTEGRATION -- see docs/PATTERNS.md §1,
//               exemplified by flagship 9.02 SEIR and 13.02 PBPK)
//   An in-silico artificial-pancreas trial simulates the SAME closed-loop system
//   (Bergman glucose-insulin ODE + PID insulin controller + meal disturbance)
//   for a whole COHORT of virtual patients that differ in physiology. Each
//   patient's simulation is sequential in time but INDEPENDENT of the others, so
//   we give each virtual patient its own GPU thread: the thread runs the full
//   RK4 + control loop (in registers) and writes one PatientResult. This is the
//   embarrassingly-parallel "one thread per patient per trajectory" mapping the
//   catalog calls for, and how RL / uncertainty studies scale these simulators.
//
//   The simulation core (simulate_patient) is shared with the CPU (bergman.h),
//   so the GPU results match the reference to round-off. kernels.cu defines the
//   kernel + host wrapper.
//
//   This header contains a __global__ declaration, so only .cu units may include
//   it -- the CPU reference uses the pure-C++ reference_cpu.h instead.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, bergman.h, reference_cpu.h.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // CohortConfig, PatientResult (pure C++, safe in .cu)

// ---- Device kernel -------------------------------------------------------
// cohort_kernel: thread `idx` simulates virtual patient idx and writes its
//   PatientResult. It builds its own PatientParams from the cohort grid via
//   patient_params() (bergman.h/reference_cpu.h), then runs the full closed-loop
//   integration -- no inter-thread communication whatsoever.
//     out : device array [cohort_size] of PatientResult (one slot per patient)
__global__ void cohort_kernel(CohortConfig c, PatientResult* __restrict__ out);

// ---- Host wrapper --------------------------------------------------------
// simulate_cohort_gpu: launch one thread per patient, copy the results back, and
//   report the measured KERNEL time (CUDA events) via *kernel_ms. main.cu calls
//   exactly this; all CUDA bookkeeping is hidden here.
//     c         : the cohort configuration (passed by value into the kernel)
//     results   : host output, resized to cohort_size (output parameter)
//     kernel_ms : out-param, milliseconds spent in the kernel itself (not copies)
void simulate_cohort_gpu(const CohortConfig& c, std::vector<PatientResult>& results,
                         float* kernel_ms);
