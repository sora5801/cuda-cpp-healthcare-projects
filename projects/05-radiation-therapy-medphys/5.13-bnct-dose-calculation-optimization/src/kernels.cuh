// ===========================================================================
// src/kernels.cuh  --  GPU Monte Carlo interface for BNCT dose
// ---------------------------------------------------------------------------
// Project 5.13 : BNCT Dose Calculation & Optimization (reduced-scope teaching MC)
//
// THE BIG IDEA (docs/PATTERNS.md §1: "stochastic / Monte-Carlo histories ->
// per-thread RNG + atomic scoring"; exemplified by flagship 5.01)
//   Neutron histories are INDEPENDENT, so each GPU thread tracks one neutron
//   (grid-stride over millions of them). Two MC-specific lessons carry over
//   directly from photon MC to neutron MC:
//     * PER-THREAD RNG: each thread seeds its own reproducible stream from its
//       history index (rng_seed in bnct_physics.h). Because that header is
//       shared with the CPU, the reference reproduces the identical histories,
//       so verification is EXACT, not statistical.
//     * ATOMIC SCORING: many threads deposit into the SAME (component, depth-bin)
//       tally cells, so the tally uses atomicAdd. Because energy is INTEGER keV
//       quanta, the atomic adds are order-independent -> the GPU result is
//       deterministic and equals the CPU tally exactly (a float dose tally would
//       NOT have this property -- float addition is not associative).
//
//   A BNCT-specific wrinkle beyond 5.01: warp divergence. Different neutrons
//   take different numbers of fast/thermal steps and hit different branches
//   (leak / scatter / capture-by-B/N/H), so threads in a warp finish at
//   different times. Production BNCT codes sort particles by material/energy
//   into batches to shrink this divergence; we keep the simple version and
//   explain the optimization in THEORY.md §GPU-mapping.
//
//   This header is included only by .cu units (it declares a __global__). The
//   CPU-only problem definition lives in reference_cpu.h.
//
// READ THIS AFTER: util/cuda_check.cuh, util/timer.cuh, bnct_physics.h,
// reference_cpu.h. Then read kernels.cu.
// ===========================================================================
#pragma once

#include <vector>
#include "reference_cpu.h"   // BnctProblem, SimParams, DoseTally (pure C++)

// ---------------------------------------------------------------------------
// dose_kernel: grid-stride over neutron histories. Each iteration simulates one
// neutron with its own reproducible RNG stream and atomically adds its deposits
// into the flattened tally. Layout of `tally`: DC_COUNT rows of n_bins columns,
// row-major, so component c depth-bin b lives at tally[c * n_bins + b].
//   grid  : a fixed large grid; the grid-stride loop covers any n_histories
//   block : 256 threads (good occupancy on sm_75..sm_89)
//   thread: history index = blockIdx.x*blockDim.x + threadIdx.x, then + stride
// ---------------------------------------------------------------------------
__global__ void dose_kernel(SimParams sp, unsigned long long n_histories,
                            unsigned long long seed, int n_bins,
                            unsigned long long* __restrict__ tally);

// ---------------------------------------------------------------------------
// dose_gpu: host wrapper. Allocates + zeros the device tally, launches all
// histories, copies the flattened tally back, and unpacks it into `t` (a
// DoseTally of DC_COUNT x n_bins). Reports the measured KERNEL time via
// *kernel_ms (CUDA events, not host clock).
//   prob      : the BNCT problem (slab + cross sections + history count + seed)
//   t         : output tally, filled to match the CPU reference exactly
//   kernel_ms : out-param, milliseconds spent in the kernel itself
// ---------------------------------------------------------------------------
void dose_gpu(const BnctProblem& prob, DoseTally& t, float* kernel_ms);
