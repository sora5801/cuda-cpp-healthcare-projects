// ===========================================================================
// src/kernels.cuh  --  GPU ensemble-perfusion interface (declarations + idea)
// ---------------------------------------------------------------------------
// Project 6.25 : Liver & Kidney Perfusion Modeling
//
// THE BIG IDEA (ensemble ODE pattern -- PATTERNS.md section 1)
//   A lobule is thousands of parallel SINUSOIDS. Each sinusoid is an independent
//   1-D convection-reaction ODE (drug carried by blood, cleared by zonal
//   Michaelis-Menten enzymes) whose only per-member difference is the inlet blood
//   VELOCITY. Because the members do not interact, we give EACH SINUSOID ITS OWN
//   GPU THREAD: the thread runs the full RK4 spatial march (perfusion.h) in
//   registers and writes one SinusoidResult. This is the natural GPU mapping for
//   organ-on-chip / virtual-pharmacology sweeps -- it scales to millions of
//   segments simply by launching more threads.
//
//   The RK4 integrator + Michaelis-Menten physics are SHARED with the CPU
//   (perfusion.h), so the GPU results match the reference to round-off.
//   kernels.cu defines the kernel and the host wrapper.
//
//   Included only by .cu translation units (it contains a __global__ decl, so the
//   plain C++ compiler must never see it -- that is why LobuleConfig lives in the
//   pure-C++ reference_cpu.h).
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, perfusion.h,
//   reference_cpu.h. Then read kernels.cu.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // LobuleConfig, SinusoidResult (pure C++, safe in .cu)

// ---- Device kernel -------------------------------------------------------
// perfusion_kernel: thread `idx` integrates sinusoid idx and writes its result.
//   It reads its inlet velocity from the sweep (sinusoid_velocity) then calls the
//   shared integrate_sinusoid(). Pure embarrassing parallelism -- no inter-thread
//   communication, no shared memory, no atomics.
//     out : device array of `lobule_size(c)` SinusoidResult, one per sinusoid.
__global__ void perfusion_kernel(LobuleConfig c, SinusoidResult* __restrict__ out);

// ---- Host wrapper --------------------------------------------------------
// integrate_gpu: launch one thread per sinusoid, copy the results back, and
//   report the measured KERNEL time (CUDA events) via *kernel_ms. main.cu calls
//   exactly this; all CUDA bookkeeping is hidden here.
//     results   : host output, resized to lobule_size(c) (output parameter)
//     kernel_ms : out-param, milliseconds spent in the kernel (not the D2H copy)
void integrate_gpu(const LobuleConfig& c, std::vector<SinusoidResult>& results, float* kernel_ms);
